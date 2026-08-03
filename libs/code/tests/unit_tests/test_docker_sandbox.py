"""Tests for the local Docker sandbox backend and provider (no Docker daemon needed)."""

from __future__ import annotations

import subprocess
from typing import Any
from unittest.mock import patch

import pytest
from deepagents.backends.protocol import ExecuteResponse

from deepagents_code.integrations.docker_sandbox import DockerSandbox
from deepagents_code.integrations.sandbox_config import SandboxConfig
from deepagents_code.integrations.sandbox_factory import (
    _DOCKER_DEFAULT_IMAGE,
    _DockerProvider,
    get_default_working_dir,
    verify_sandbox_deps,
)
from deepagents_code.integrations.sandbox_provider import SandboxNotFoundError
from deepagents_code.integrations.sandbox_registry import SandboxRegistry

_BACKEND = "deepagents_code.integrations.docker_sandbox"
_FACTORY = "deepagents_code.integrations.sandbox_factory"


def _completed(
    args: list[str],
    returncode: int = 0,
    stdout: str | bytes = "",
    stderr: str | bytes = "",
) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=args, returncode=returncode, stdout=stdout, stderr=stderr
    )


# ---------------------------------------------------------------------------
# DockerSandbox backend
# ---------------------------------------------------------------------------


class TestDockerSandboxExecute:
    """Command execution via `docker exec`."""

    def test_execute_builds_docker_exec_argv_and_maps_result(self) -> None:
        """Commands run through `docker exec -w <wd> <id> /bin/sh -c`."""
        backend = DockerSandbox("cid123", working_dir="/workspace")
        with patch(f"{_BACKEND}.subprocess.run") as run:
            run.return_value = _completed([], returncode=0, stdout=b"hello\n")
            result = backend.execute("echo hello")

        argv = run.call_args.args[0]
        assert argv[:4] == ["docker", "exec", "-w", "/workspace"]
        assert argv[4:] == ["cid123", "/bin/sh", "-c", "echo hello"]
        assert result.output == "hello\n"
        assert result.exit_code == 0
        assert result.truncated is False

    def test_execute_nonzero_exit_code_preserved(self) -> None:
        """`docker exec` failures surface as non-zero exit codes with output."""
        backend = DockerSandbox("cid123")
        with patch(f"{_BACKEND}.subprocess.run") as run:
            run.return_value = _completed(
                [], returncode=126, stdout=b"container not running"
            )
            result = backend.execute("echo hi")
        assert result.exit_code == 126
        assert "container not running" in result.output

    def test_execute_timeout_returns_124(self) -> None:
        """A timed-out command returns the standard timeout exit code."""
        backend = DockerSandbox("cid123")
        with patch(f"{_BACKEND}.subprocess.run") as run:
            run.side_effect = subprocess.TimeoutExpired(cmd="docker", timeout=5)
            result = backend.execute("sleep 100", timeout=5)
        assert result.exit_code == 124
        assert "timed out after 5 seconds" in result.output

    def test_execute_empty_command_is_an_error(self) -> None:
        """Empty commands are rejected without invoking docker."""
        backend = DockerSandbox("cid123")
        with patch(f"{_BACKEND}.subprocess.run") as run:
            result = backend.execute("")
        run.assert_not_called()
        assert result.exit_code == 1

    def test_execute_truncates_oversized_output(self) -> None:
        """Output beyond `max_output_bytes` is truncated and flagged."""
        backend = DockerSandbox("cid123", max_output_bytes=10)
        with patch(f"{_BACKEND}.subprocess.run") as run:
            run.return_value = _completed([], returncode=0, stdout=b"x" * 100)
            result = backend.execute("yes")
        assert result.truncated is True
        assert "Output truncated at 10 bytes" in result.output

    def test_invalid_timeouts_raise(self) -> None:
        """Non-positive timeouts are rejected at init and per call."""
        with pytest.raises(ValueError, match="timeout must be positive"):
            DockerSandbox("cid123", timeout=0)
        backend = DockerSandbox("cid123")
        with pytest.raises(ValueError, match="timeout must be positive"):
            backend.execute("echo hi", timeout=-1)

    def test_id_and_working_dir_properties(self) -> None:
        """`id` returns the container id; `working_dir` the exec directory."""
        backend = DockerSandbox("cid123", working_dir="/srv")
        assert backend.id == "cid123"
        assert backend.working_dir == "/srv"


class TestDockerSandboxUpload:
    """File upload via `docker exec -i ... cat`."""

    def test_upload_streams_content_over_stdin(self) -> None:
        """Content bytes are piped to a `mkdir -p && cat` script."""
        backend = DockerSandbox("cid123")
        with patch(f"{_BACKEND}.subprocess.run") as run:
            run.return_value = _completed([], returncode=0)
            responses = backend.upload_files([("/tmp/a.txt", b"data")])

        argv = run.call_args.args[0]
        assert argv[:4] == ["docker", "exec", "-i", "cid123"]
        assert "cat >" in argv[-1]
        assert run.call_args.kwargs["input"] == b"data"
        assert responses[0].path == "/tmp/a.txt"
        assert responses[0].error is None

    def test_upload_relative_path_rejected_without_docker_call(self) -> None:
        """Relative paths map to `invalid_path` and never reach docker."""
        backend = DockerSandbox("cid123")
        with patch(f"{_BACKEND}.subprocess.run") as run:
            responses = backend.upload_files([("relative.txt", b"data")])
        run.assert_not_called()
        assert responses[0].error == "invalid_path"

    def test_upload_maps_permission_denied(self) -> None:
        """A permission failure maps to the standardized error literal."""
        backend = DockerSandbox("cid123")
        with patch(f"{_BACKEND}.subprocess.run") as run:
            run.return_value = _completed(
                [], returncode=1, stderr=b"/bin/sh: 1: cannot create: Permission denied"
            )
            responses = backend.upload_files([("/etc/ro.txt", b"data")])
        assert responses[0].error == "permission_denied"

    def test_upload_partial_success_preserves_order(self) -> None:
        """Each file gets its own response, in input order."""
        backend = DockerSandbox("cid123")
        with patch(f"{_BACKEND}.subprocess.run") as run:
            run.return_value = _completed([], returncode=0)
            responses = backend.upload_files([("bad.txt", b"1"), ("/ok.txt", b"2")])
        assert [r.path for r in responses] == ["bad.txt", "/ok.txt"]
        assert responses[0].error == "invalid_path"
        assert responses[1].error is None


class TestDockerSandboxDownload:
    """File download via `docker exec ... cat` with a classifying probe."""

    def test_download_returns_bytes(self) -> None:
        """A readable file's raw bytes are returned."""
        backend = DockerSandbox("cid123")
        with patch(f"{_BACKEND}.subprocess.run") as run:
            run.return_value = _completed([], returncode=0, stdout=b"\x00binary\xff")
            responses = backend.download_files(["/tmp/a.bin"])
        assert responses[0].content == b"\x00binary\xff"
        assert responses[0].error is None

    @pytest.mark.parametrize(
        ("exit_code", "expected_error"),
        [
            (64, "file_not_found"),
            (65, "is_directory"),
            (66, "permission_denied"),
        ],
    )
    def test_download_maps_probe_exit_codes(
        self, exit_code: int, expected_error: str
    ) -> None:
        """The probe script's sentinel exit codes map to standard errors."""
        backend = DockerSandbox("cid123")
        with patch(f"{_BACKEND}.subprocess.run") as run:
            run.return_value = _completed([], returncode=exit_code, stdout=b"")
            responses = backend.download_files(["/missing"])
        assert responses[0].error == expected_error
        assert responses[0].content is None

    def test_download_relative_path_rejected(self) -> None:
        """Relative paths map to `invalid_path` without invoking docker."""
        backend = DockerSandbox("cid123")
        with patch(f"{_BACKEND}.subprocess.run") as run:
            responses = backend.download_files(["relative.txt"])
        run.assert_not_called()
        assert responses[0].error == "invalid_path"

    def test_download_unknown_failure_surfaces_stderr(self) -> None:
        """Unclassified failures return the docker CLI's stderr text."""
        backend = DockerSandbox("cid123")
        with patch(f"{_BACKEND}.subprocess.run") as run:
            run.return_value = _completed(
                [], returncode=125, stderr=b"Error: no such container"
            )
            responses = backend.download_files(["/tmp/a"])
        assert responses[0].error == "Error: no such container"


# ---------------------------------------------------------------------------
# _DockerProvider
# ---------------------------------------------------------------------------


def _provider_with_daemon() -> _DockerProvider:
    """Construct a provider with the CLI/daemon preflight mocked green."""
    with (
        patch(f"{_FACTORY}.shutil.which", return_value="/usr/bin/docker"),
        patch(f"{_FACTORY}.subprocess.run") as run,
    ):
        run.return_value = _completed([], returncode=0, stdout="29.0.0\n")
        return _DockerProvider()


class TestDockerProviderInit:
    """CLI/daemon preflight checks."""

    def test_missing_cli_raises(self) -> None:
        """A missing `docker` executable produces an actionable error."""
        with (
            patch(f"{_FACTORY}.shutil.which", return_value=None),
            pytest.raises(ValueError, match="Docker CLI not found"),
        ):
            _DockerProvider()

    def test_unreachable_daemon_raises(self) -> None:
        """A dead daemon produces an actionable error."""
        probe_failure = _completed(
            [], returncode=1, stderr="Cannot connect to the Docker daemon"
        )
        with (
            patch(f"{_FACTORY}.shutil.which", return_value="/usr/bin/docker"),
            patch(f"{_FACTORY}.subprocess.run", return_value=probe_failure),
            pytest.raises(ValueError, match="Docker daemon is not reachable"),
        ):
            _DockerProvider()


class TestDockerProviderGetOrCreate:
    """Container creation and attach."""

    def test_create_runs_container_with_default_image(self) -> None:
        """A fresh sandbox is a detached container with label and workdir."""
        provider = _provider_with_daemon()
        with (
            patch(f"{_FACTORY}.subprocess.run") as run,
            patch.object(
                DockerSandbox,
                "execute",
                return_value=ExecuteResponse(output="ready\n", exit_code=0),
            ),
        ):
            run.return_value = _completed([], returncode=0, stdout="cid456\n")
            backend = provider.get_or_create()

        argv = run.call_args.args[0]
        assert argv[1] == "run"
        assert "-d" in argv
        assert _DOCKER_DEFAULT_IMAGE in argv
        assert "/workspace" in argv
        assert any(a.startswith("com.deepagents.code.sandbox") for a in argv)
        assert backend.id == "cid456"

    def test_create_honors_image_param_and_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit `image` wins over env var; env var wins over default."""
        provider = _provider_with_daemon()
        monkeypatch.delenv("DEEPAGENTS_CODE_DOCKER_SANDBOX_IMAGE", raising=False)
        monkeypatch.setenv("DOCKER_SANDBOX_IMAGE", "custom:from-env")
        with (
            patch(f"{_FACTORY}.subprocess.run") as run,
            patch.object(
                DockerSandbox,
                "execute",
                return_value=ExecuteResponse(output="ready\n", exit_code=0),
            ),
        ):
            run.return_value = _completed([], returncode=0, stdout="cid1\n")
            provider.get_or_create()
            assert "custom:from-env" in run.call_args.args[0]

            provider.get_or_create(image="custom:explicit")
            assert "custom:explicit" in run.call_args.args[0]

    def test_create_failure_raises_runtime_error(self) -> None:
        """`docker run` failures propagate with stderr detail."""
        provider = _provider_with_daemon()
        with patch(f"{_FACTORY}.subprocess.run") as run:
            run.return_value = _completed(
                [], returncode=125, stderr="pull access denied"
            )
            with pytest.raises(RuntimeError, match="pull access denied"):
                provider.get_or_create()

    def test_create_cleans_up_when_never_ready(self) -> None:
        """A container that never answers the readiness poll is removed."""
        provider = _provider_with_daemon()
        with (
            patch(f"{_FACTORY}.subprocess.run") as run,
            patch(f"{_FACTORY}.time.sleep"),
            patch.object(
                DockerSandbox,
                "execute",
                return_value=ExecuteResponse(output="", exit_code=1),
            ),
        ):
            run.return_value = _completed([], returncode=0, stdout="cid456\n")
            with pytest.raises(RuntimeError, match="failed to become ready"):
                provider.get_or_create(timeout=4)
        # Last CLI call should be the cleanup `docker rm -f`.
        assert run.call_args.args[0][1] == "rm"

    def test_unknown_kwargs_rejected(self) -> None:
        """Unsupported keyword arguments raise `TypeError`."""
        provider = _provider_with_daemon()
        with pytest.raises(TypeError, match="snapshot"):
            provider.get_or_create(snapshot="nope")

    def test_attach_missing_container_raises_not_found(self) -> None:
        """Attaching to an unknown container raises `SandboxNotFoundError`."""
        provider = _provider_with_daemon()
        with patch(f"{_FACTORY}.subprocess.run") as run:
            run.return_value = _completed(
                [], returncode=1, stderr="Error: No such object"
            )
            with pytest.raises(SandboxNotFoundError):
                provider.get_or_create(sandbox_id="ghost")

    def test_attach_starts_stopped_container(self) -> None:
        """A stopped container is started before attaching."""
        provider = _provider_with_daemon()
        calls: list[list[str]] = []

        def fake_run(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess:
            calls.append(argv)
            if argv[1] == "inspect":
                return _completed(argv, returncode=0, stdout="exited\n")
            return _completed(argv, returncode=0, stdout="")

        with patch(f"{_FACTORY}.subprocess.run", side_effect=fake_run):
            backend = provider.get_or_create(sandbox_id="cid789")

        assert backend.id == "cid789"
        subcommands = [argv[1] for argv in calls]
        assert subcommands == ["inspect", "start", "exec"]
        # The exec call prepares the working directory.
        assert calls[-1][2:] == ["cid789", "mkdir", "-p", "/workspace"]

    def test_attach_running_container_skips_start(self) -> None:
        """A running container is attached without `docker start`."""
        provider = _provider_with_daemon()
        calls: list[list[str]] = []

        def fake_run(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess:
            calls.append(argv)
            if argv[1] == "inspect":
                return _completed(argv, returncode=0, stdout="running\n")
            return _completed(argv, returncode=0, stdout="")

        with patch(f"{_FACTORY}.subprocess.run", side_effect=fake_run):
            provider.get_or_create(sandbox_id="cid789")

        assert [argv[1] for argv in calls] == ["inspect", "exec"]


class TestDockerProviderDelete:
    """Container removal."""

    def test_delete_force_removes(self) -> None:
        """Delete issues `docker rm -f <id>`."""
        provider = _provider_with_daemon()
        with patch(f"{_FACTORY}.subprocess.run") as run:
            run.return_value = _completed([], returncode=0)
            provider.delete(sandbox_id="cid456")
        assert run.call_args.args[0][1:] == ["rm", "-f", "cid456"]

    def test_delete_missing_container_is_noop(self) -> None:
        """Removing an already-removed container does not raise."""
        provider = _provider_with_daemon()
        with patch(f"{_FACTORY}.subprocess.run") as run:
            run.return_value = _completed(
                [], returncode=1, stderr="Error: No such container: cid456"
            )
            provider.delete(sandbox_id="cid456")

    def test_delete_other_failure_raises(self) -> None:
        """Non-missing-container failures propagate."""
        provider = _provider_with_daemon()
        with patch(f"{_FACTORY}.subprocess.run") as run:
            run.return_value = _completed([], returncode=1, stderr="daemon exploded")
            with pytest.raises(RuntimeError, match="daemon exploded"):
                provider.delete(sandbox_id="cid456")


# ---------------------------------------------------------------------------
# Registry / factory wiring
# ---------------------------------------------------------------------------


class TestDockerRegistration:
    """The `docker` provider is a first-class built-in."""

    def test_registry_knows_docker(self) -> None:
        """`docker` is available with the expected metadata."""
        registry = SandboxRegistry(config=SandboxConfig(), include_entry_points=False)
        assert registry.is_available("docker")
        metadata = registry.get_metadata("docker")
        assert metadata is not None
        assert metadata.working_dir == "/workspace"
        assert metadata.supports_sandbox_id is True
        assert metadata.supports_snapshot_name is False
        # Bundled: no extra to install, no backend module to probe.
        assert metadata.install is None
        assert metadata.backend_module is None

    def test_default_working_dir(self) -> None:
        """The factory resolves the docker working dir through the registry."""
        assert get_default_working_dir("docker") == "/workspace"

    def test_verify_sandbox_deps_is_a_noop(self) -> None:
        """No optional dependency is required for the docker provider."""
        verify_sandbox_deps("docker")  # should not raise

    def test_create_provider_returns_docker_provider(self) -> None:
        """The registry constructs `_DockerProvider` for `docker`."""
        registry = SandboxRegistry(config=SandboxConfig(), include_entry_points=False)
        with (
            patch(f"{_FACTORY}.shutil.which", return_value="/usr/bin/docker"),
            patch(f"{_FACTORY}.subprocess.run") as run,
        ):
            run.return_value = _completed([], returncode=0, stdout="29.0.0\n")
            provider = registry.create_provider("docker")
        assert isinstance(provider, _DockerProvider)
