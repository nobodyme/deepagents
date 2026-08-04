"""Local Docker sandbox backend.

[`DockerSandbox`][deepagents_code.integrations.docker_sandbox.DockerSandbox]
wraps a running Docker container and implements the sandbox backend contract by
shelling out to the `docker` CLI via `subprocess` — no Docker SDK dependency.
Container lifecycle (create / attach / delete) is handled by the built-in
`docker` provider in
[`sandbox_factory`][deepagents_code.integrations.sandbox_factory].

Unlike the remote providers (Daytona, Modal, Runloop, ...), this backend runs
containers on the *local* Docker daemon, making it suitable for offline
development and testing without any cloud account or API key. Isolation is
whatever the local Docker daemon provides — containers share the host kernel,
so this is weaker isolation than a remote microVM sandbox.
"""

from __future__ import annotations

import shlex
import subprocess  # noqa: S404  # fixed-argv `docker` CLI calls; commands run inside the container
from typing import Final

from deepagents.backends.protocol import (
    FILE_NOT_FOUND,
    INVALID_PATH,
    IS_DIRECTORY,
    PERMISSION_DENIED,
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

DEFAULT_EXECUTE_TIMEOUT: Final = 30 * 60
"""Default timeout in seconds for command execution (matches `DaytonaSandbox`)."""

MAX_OUTPUT_BYTES: Final = 5 * 1024 * 1024
"""Cap on captured command output (5 MiB).

Must stay comfortably above the ~500 KiB pages that `BaseSandbox`'s
server-side read/grep scripts emit, or their JSON output would be cut
mid-payload and fail to parse.
"""

_DOCKER_CLI_TIMEOUT_MARGIN: Final = 10
"""Extra seconds granted to the `docker` CLI process beyond the command timeout.

`docker exec` adds client/daemon round-trip overhead on top of the command's
own runtime; without headroom a command finishing right at the timeout would be
misreported as timed out.
"""

# Exit codes used by the download probe script to classify failures without
# parsing localized error text. Chosen from the 64-113 user-defined range so
# they cannot collide with `cat`'s own exit codes (1) or shell/`docker` codes
# (125-127).
_EXIT_NOT_FOUND: Final = 64
_EXIT_IS_DIRECTORY: Final = 65
_EXIT_PERMISSION_DENIED: Final = 66
_EXIT_NOT_A_REGULAR_FILE: Final = 67

_DOCKER_CLI_ERROR_MARKERS: Final = (
    "error response from daemon",
    "cannot connect to the docker daemon",
    "error during connect",
)
"""Lowercase stderr markers identifying a `docker` client/daemon failure.

Used to tell a *transport* failure (container stopped or removed, daemon
unreachable) apart from an ordinary non-zero exit of the command running
inside the container. Transport failures must raise rather than return an
`ExecuteResponse`: `BaseSandbox`'s `ls`/`glob` parsers ignore exit codes and
would otherwise read the docker error text as a successful empty listing.
"""


class DockerCliError(RuntimeError):
    """The `docker` CLI itself failed — the sandbox is unreachable.

    Raised when `docker exec` cannot reach the container at all (container
    stopped or removed, daemon down), as opposed to the executed command
    failing inside a healthy container. Mirrors how remote backends surface
    SDK transport exceptions instead of fabricating a command result.
    """


class DockerSandbox(BaseSandbox):
    """Sandbox backend backed by a running local Docker container.

    Implements `execute()`, `upload_files()`, and `download_files()` on top of
    the `docker` CLI; all other file operations are inherited from
    [`BaseSandbox`][deepagents.backends.sandbox.BaseSandbox], which requires
    `python3` and a POSIX shell inside the container image.

    The backend does not manage the container lifecycle — it assumes
    `container_id` names a container that is already running. Use the `docker`
    sandbox provider (`create_sandbox("docker")`) to create and clean up
    containers.
    """

    def __init__(
        self,
        container_id: str,
        *,
        working_dir: str = "/workspace",
        timeout: int = DEFAULT_EXECUTE_TIMEOUT,
        max_output_bytes: int = MAX_OUTPUT_BYTES,
        docker_cli: str = "docker",
    ) -> None:
        """Wrap an existing running container.

        Args:
            container_id: Docker container ID or name to execute in.
            working_dir: Working directory for executed commands. Must exist in
                the container (the provider creates it).
            timeout: Default command timeout in seconds used when `execute()`
                is called without an explicit `timeout`.
            max_output_bytes: Cap on captured command output; longer output is
                truncated and flagged.
            docker_cli: Path or name of the `docker` executable.

        Raises:
            ValueError: If `timeout` is not positive.
        """
        if timeout <= 0:
            msg = f"timeout must be positive, got {timeout}"
            raise ValueError(msg)
        self._container_id = container_id
        self._working_dir = working_dir
        self._default_timeout = timeout
        self._max_output_bytes = max_output_bytes
        self._docker_cli = docker_cli

    @property
    def id(self) -> str:
        """The Docker container ID (or name) this backend executes in."""
        return self._container_id

    @property
    def working_dir(self) -> str:
        """Working directory used for command execution."""
        return self._working_dir

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command inside the container via `docker exec`.

        The command's stderr is appended to stdout in the returned output
        (wrapped in a `<stderr>` block, matching `DaytonaSandbox`).
        `BaseSandbox`'s helper scripts redirect with `2>&1` inside the
        container, so their JSON output is unaffected. Keeping the client-side
        streams separate is what lets a `docker` transport failure be
        distinguished from the command itself failing.

        Args:
            command: Shell command string, run with `/bin/sh -c`.
            timeout: Maximum seconds to wait. `None` uses the default from
                `__init__`. On expiry the local `docker exec` client process is
                killed; the command inside the container may keep running.

        Returns:
            `ExecuteResponse` with output, exit code, and truncation flag.

        Raises:
            ValueError: If `timeout` is not positive.
            DockerCliError: If the container is unreachable (stopped, removed,
                or the daemon is down) rather than the command failing.
        """
        if not command or not isinstance(command, str):
            return ExecuteResponse(
                output="Error: Command must be a non-empty string.",
                exit_code=1,
            )
        effective_timeout = timeout if timeout is not None else self._default_timeout
        if effective_timeout <= 0:
            msg = f"timeout must be positive, got {effective_timeout}"
            raise ValueError(msg)

        try:
            proc = subprocess.run(  # noqa: S603  # fixed argv; command runs inside the container, which is the point
                [
                    self._docker_cli,
                    "exec",
                    "-w",
                    self._working_dir,
                    self._container_id,
                    "/bin/sh",
                    "-c",
                    command,
                ],
                capture_output=True,
                stdin=subprocess.DEVNULL,
                timeout=effective_timeout + _DOCKER_CLI_TIMEOUT_MARGIN,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ExecuteResponse(
                output=(
                    f"Error: Command timed out after {effective_timeout} seconds. "
                    "For long-running commands, re-run using the timeout "
                    "parameter. The process may still be running inside the "
                    "container."
                ),
                exit_code=124,  # Standard timeout exit code
            )
        except OSError as e:
            msg = f"Failed to invoke the docker CLI: {e}"
            raise DockerCliError(msg) from e

        if proc.returncode != 0 and not proc.stdout:
            stderr_text = proc.stderr.decode("utf-8", errors="replace").strip()
            lowered = stderr_text.lower()
            if any(marker in lowered for marker in _DOCKER_CLI_ERROR_MARKERS):
                msg = (
                    f"Docker sandbox container '{self._container_id}' is "
                    f"unreachable: {stderr_text}"
                )
                raise DockerCliError(msg)

        raw = proc.stdout
        if proc.stderr:
            stderr_body = proc.stderr.strip()
            raw += b"\n<stderr>" + stderr_body + b"</stderr>"
        truncated = False
        if len(raw) > self._max_output_bytes:
            raw = raw[: self._max_output_bytes]
            truncated = True
        output = raw.decode("utf-8", errors="replace")
        if truncated:
            output += f"\n\n... Output truncated at {self._max_output_bytes} bytes."

        return ExecuteResponse(
            output=output,
            exit_code=proc.returncode,
            truncated=truncated,
        )

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files into the container by streaming bytes over `docker exec` stdin.

        Parent directories are created as needed. Each file is transferred
        independently so partial success is reported per file.

        Args:
            files: List of `(absolute_path, content)` tuples.

        Returns:
            One `FileUploadResponse` per input, in input order.
        """
        responses: list[FileUploadResponse] = []
        for path, content in files:
            if not path.startswith("/"):
                responses.append(FileUploadResponse(path=path, error=INVALID_PATH))
                continue
            quoted = shlex.quote(path)
            script = f'p={quoted}; d=$(dirname "$p"); mkdir -p "$d" && cat > "$p"'
            try:
                proc = subprocess.run(  # noqa: S603  # fixed argv; path is shell-quoted
                    [
                        self._docker_cli,
                        "exec",
                        "-i",
                        self._container_id,
                        "/bin/sh",
                        "-c",
                        script,
                    ],
                    input=content,
                    capture_output=True,
                    timeout=self._default_timeout,
                    check=False,
                )
            except (subprocess.TimeoutExpired, OSError) as e:
                responses.append(
                    FileUploadResponse(path=path, error=f"upload failed: {e}")
                )
                continue
            if proc.returncode != 0:
                stderr = proc.stderr.decode("utf-8", errors="replace").strip()
                responses.append(
                    FileUploadResponse(path=path, error=_map_write_error(stderr))
                )
            else:
                responses.append(FileUploadResponse(path=path))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from the container by streaming `cat` output.

        A probe script classifies missing paths, directories, non-regular
        files (FIFOs, device nodes — which `cat` would block on or stream
        unboundedly), and unreadable files via distinct exit codes so failures
        map to the standardized `FileOperationError` literals.

        Args:
            paths: Absolute paths to download.

        Returns:
            One `FileDownloadResponse` per input, in input order.
        """
        responses: list[FileDownloadResponse] = []
        for path in paths:
            if not path.startswith("/"):
                responses.append(FileDownloadResponse(path=path, error=INVALID_PATH))
                continue
            quoted = shlex.quote(path)
            script = (
                f"p={quoted}; "
                f'if [ -d "$p" ]; then exit {_EXIT_IS_DIRECTORY}; '
                f'elif [ ! -e "$p" ]; then exit {_EXIT_NOT_FOUND}; '
                f'elif [ ! -f "$p" ]; then exit {_EXIT_NOT_A_REGULAR_FILE}; '
                f'elif [ ! -r "$p" ]; then exit {_EXIT_PERMISSION_DENIED}; '
                f'else exec cat "$p"; fi'
            )
            try:
                proc = subprocess.run(  # noqa: S603  # fixed argv; path is shell-quoted
                    [
                        self._docker_cli,
                        "exec",
                        self._container_id,
                        "/bin/sh",
                        "-c",
                        script,
                    ],
                    capture_output=True,
                    timeout=self._default_timeout,
                    check=False,
                )
            except (subprocess.TimeoutExpired, OSError) as e:
                responses.append(
                    FileDownloadResponse(path=path, error=f"download failed: {e}")
                )
                continue
            if proc.returncode == 0:
                responses.append(FileDownloadResponse(path=path, content=proc.stdout))
            elif proc.returncode == _EXIT_NOT_FOUND:
                responses.append(FileDownloadResponse(path=path, error=FILE_NOT_FOUND))
            elif proc.returncode == _EXIT_IS_DIRECTORY:
                responses.append(FileDownloadResponse(path=path, error=IS_DIRECTORY))
            elif proc.returncode == _EXIT_NOT_A_REGULAR_FILE:
                responses.append(
                    FileDownloadResponse(path=path, error="not a regular file")
                )
            elif proc.returncode == _EXIT_PERMISSION_DENIED:
                responses.append(
                    FileDownloadResponse(path=path, error=PERMISSION_DENIED)
                )
            else:
                stderr = proc.stderr.decode("utf-8", errors="replace").strip()
                responses.append(
                    FileDownloadResponse(
                        path=path,
                        error=stderr
                        or f"download failed with exit code {proc.returncode}",
                    )
                )
        return responses


def _map_write_error(stderr: str) -> str:
    """Map a shell/`cat` error message to a standardized error when possible.

    Args:
        stderr: Captured stderr from the failed upload script.

    Returns:
        A `FileOperationError` literal for recognized conditions, or the raw
        stderr text (never empty) otherwise.
    """
    lowered = stderr.lower()
    if "permission denied" in lowered:
        return PERMISSION_DENIED
    if "is a directory" in lowered:
        return IS_DIRECTORY
    return stderr or "upload failed"


__all__ = [
    "DEFAULT_EXECUTE_TIMEOUT",
    "MAX_OUTPUT_BYTES",
    "DockerCliError",
    "DockerSandbox",
]
