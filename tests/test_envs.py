"""Tests for environment registry, dispatch, and setup command construction."""

from __future__ import annotations

import subprocess
import sys

import pytest

from argo_deepmsi import envs


class _Execed(Exception):
    pass


@pytest.fixture
def record_exec(monkeypatch):
    calls: list[tuple[str, list[str], dict[str, str]]] = []

    def fake_exec(file: str, argv: list[str], env: dict[str, str]) -> None:
        calls.append((file, argv, env))
        raise _Execed

    monkeypatch.setattr(envs, "_exec", fake_exec)
    monkeypatch.delenv("ARGO_ENV", raising=False)
    monkeypatch.delenv("ARGO_NO_DISPATCH", raising=False)
    return calls


def test_env_command_shapes():
    lazy = envs.ENVS["lazyslide"].project_dir
    assert envs.env_command("lazyslide", ["argo", "models"]) == [
        "uv",
        "run",
        "--frozen",
        "--project",
        str(lazy),
        "argo",
        "models",
    ]
    assert envs.env_command("mussel", []) == [
        "uv",
        "run",
        "--frozen",
        "--project",
        str(envs.ENVS_DIR / "mussel"),
    ]
    assert envs.env_command("core", ["argo"])[4] == str(envs.REPO_ROOT)
    bin_dir = envs.ENVS_DIR / "paladin" / ".venv" / "bin"
    assert envs.env_command("paladin", ["python", "-V"]) == [str(bin_dir / "python"), "-V"]
    assert envs.env_command("paladin", []) == [str(bin_dir / "python")]
    with pytest.raises(KeyError):
        envs.env_command("nope", [])


def test_module_available_does_not_import():
    assert envs.module_available("json")
    assert not envs.module_available("definitely_not_a_module_xyz")
    assert not envs.module_available("definitely_not_a_pkg_xyz.sub")


def test_ensure_env_returns_when_probe_importable(monkeypatch, record_exec):
    monkeypatch.setattr(envs, "module_available", lambda module: True)
    envs.ensure_env("lazyslide")
    assert record_exec == []


def test_ensure_env_reexecs_with_same_argv(monkeypatch, record_exec, tmp_path):
    project = tmp_path / "lazyslide"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='x'\n")
    spec = envs.EnvSpec("lazyslide", project, "uv", "lazyslide", "test")
    monkeypatch.setitem(envs.ENVS, "lazyslide", spec)
    monkeypatch.setattr(envs, "module_available", lambda module: False)
    monkeypatch.setattr(sys, "argv", ["/core/bin/argo", "aggregate", "uni2", "--method", "mean"])
    monkeypatch.setenv("VIRTUAL_ENV", "/core/.venv")

    with pytest.raises(_Execed):
        envs.ensure_env("lazyslide")

    ((file, argv, env),) = record_exec
    assert file == "uv"
    assert argv == [
        "uv",
        "run",
        "--frozen",
        "--project",
        str(project),
        "argo",
        "aggregate",
        "uni2",
        "--method",
        "mean",
    ]
    assert env["ARGO_ENV"] == "lazyslide"
    assert "VIRTUAL_ENV" not in env


def test_ensure_env_loop_guard(monkeypatch, record_exec):
    monkeypatch.setattr(envs, "module_available", lambda module: False)
    monkeypatch.setenv("ARGO_ENV", "lazyslide")
    with pytest.raises(envs.EnvDispatchError, match="argo setup --env lazyslide"):
        envs.ensure_env("lazyslide")
    assert record_exec == []


def test_ensure_env_no_dispatch(monkeypatch, record_exec):
    monkeypatch.setattr(envs, "module_available", lambda module: False)
    monkeypatch.setenv("ARGO_NO_DISPATCH", "1")
    with pytest.raises(envs.EnvDispatchError, match="ARGO_NO_DISPATCH"):
        envs.ensure_env("lazyslide")
    assert record_exec == []


def test_ensure_env_missing_project_does_not_exec(monkeypatch, record_exec, tmp_path):
    spec = envs.EnvSpec("lazyslide", tmp_path / "absent", "uv", "lazyslide", "test")
    monkeypatch.setitem(envs.ENVS, "lazyslide", spec)
    monkeypatch.setattr(envs, "module_available", lambda module: False)
    with pytest.raises(envs.EnvDispatchError, match="not a uv project"):
        envs.ensure_env("lazyslide")
    assert record_exec == []


def test_setup_dry_run_commands():
    assert envs.setup_env("core", dry_run=True) == [
        ["uv", "sync", "--frozen", "--project", str(envs.REPO_ROOT), "--extra", "dev"]
    ]
    assert envs.setup_env("core", extras=("dask", "dev"), dry_run=True)[0][-4:] == [
        "--extra",
        "dev",
        "--extra",
        "dask",
    ]
    assert envs.setup_env("mussel", dry_run=True) == [
        ["uv", "sync", "--frozen", "--project", str(envs.ENVS_DIR / "mussel")]
    ]
    assert envs.setup_env("paladin", dry_run=True) == [
        ["bash", str(envs.ENVS_DIR / "paladin" / "setup.sh")]
    ]


def test_setup_env_runs_and_reports_failure(monkeypatch, tmp_path):
    project = tmp_path / "mussel"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='x'\n")
    monkeypatch.setitem(
        envs.ENVS, "mussel", envs.EnvSpec("mussel", project, "uv", "mussel", "test")
    )
    seen = []

    def ok(cmd, **kwargs):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    assert envs.setup_env("mussel", runner=ok) == seen

    def fail(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 2)

    with pytest.raises(envs.EnvSetupError, match="exited with 2"):
        envs.setup_env("mussel", runner=fail)

    monkeypatch.setitem(
        envs.ENVS, "mussel", envs.EnvSpec("mussel", tmp_path / "gone", "uv", "mussel", "t")
    )
    with pytest.raises(envs.EnvSetupError, match="does not exist"):
        envs.setup_env("mussel", runner=ok)


def test_env_status_missing_project(monkeypatch, tmp_path):
    monkeypatch.setitem(
        envs.ENVS, "mussel", envs.EnvSpec("mussel", tmp_path / "gone", "uv", "mussel", "t")
    )
    status = envs.env_status("mussel")
    assert status["exists"] is False
    assert status["venv"] is False
    assert status["probe"] is None
    assert "missing" in status["detail"]


def test_dotenv_keys_reports_names_only(tmp_path):
    path = tmp_path / ".env"
    path.write_text("# c\nREDCAP_API_TOKEN=secret\nexport REDCAP_API_URL='x'\nEMPTY=\n")
    assert envs.dotenv_keys(path) == {"REDCAP_API_TOKEN", "REDCAP_API_URL"}
    assert envs.dotenv_keys(tmp_path / "missing") == set()


def test_hf_token_source_never_returns_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    assert envs.hf_token_source() == "$HF_TOKEN"


def test_repo_root_override(monkeypatch, tmp_path):
    monkeypatch.setenv("ARGO_REPO_ROOT", str(tmp_path))
    assert envs._repo_root() == tmp_path.resolve()
