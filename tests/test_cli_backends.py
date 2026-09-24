"""CLI tests for environment commands, backend dispatch, and backend comparison."""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from argo_deepmsi import cli, envs

runner = CliRunner()


def _fake_backends(monkeypatch, **modules: dict) -> None:
    """Install fake ``argo_deepmsi.backends.<name>`` modules for one test."""
    package = types.ModuleType("argo_deepmsi.backends")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "argo_deepmsi.backends", package)
    for name, attributes in modules.items():
        module = types.ModuleType(f"argo_deepmsi.backends.{name}")
        for key, value in attributes.items():
            setattr(module, key, value)
        setattr(package, name, module)
        monkeypatch.setitem(sys.modules, f"argo_deepmsi.backends.{name}", module)


def _slide_table(tmp_path: Path) -> Path:
    rows = [
        {"PATIENT": f"P{i}", "FILENAME": str(tmp_path / f"s{i}.svs"), "SITE": site}
        for i, site in enumerate(["UITH", "UITH", "UITH", "OAUTHC", "OAUTHC", "MSK"])
    ]
    path = tmp_path / "slides.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_cli_import_is_core_only():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, argo_deepmsi.cli; "
            "heavy = {'lazyslide', 'wsidata', 'scanpy', 'transformers', 'torch'}; "
            "assert not heavy & set(sys.modules), heavy & set(sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_help_lists_backend_commands():
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    for command in ("setup", "envs", "compare-backends", "paladin", "extract"):
        assert command in result.output


def test_envs_command_runs():
    result = runner.invoke(cli.app, ["envs", "--no-probe"])
    assert result.exit_code == 0, result.output
    for name in envs.ENVS:
        assert name in result.output


def test_setup_dry_run_prints_commands(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    result = runner.invoke(cli.app, ["setup", "--dry-run", "--env", "mussel", "--env", "core"])
    assert result.exit_code == 0, result.output
    assert "uv sync --frozen --project" in result.output
    assert "hf_secret_value" not in result.output
    assert "REDCAP_API_TOKEN" in result.output


def test_setup_fails_when_env_sync_fails(monkeypatch):
    def boom(name, **kwargs):
        raise envs.EnvSetupError(f"{name}: broken")

    monkeypatch.setattr(envs, "setup_env", boom)
    result = runner.invoke(cli.app, ["setup", "--env", "mussel", "--no-hf-check"])
    assert result.exit_code == 1
    assert "mussel: broken" in result.output


def test_setup_rejects_unknown_env():
    result = runner.invoke(cli.app, ["setup", "--dry-run", "--env", "nope"])
    assert result.exit_code != 0


def test_extract_mussel_uses_mussel_prefix(monkeypatch, tmp_path):
    table = _slide_table(tmp_path)
    config = tmp_path / "mussel-hoptimus0.toml"
    config.write_text("")
    calls = {}

    def run_table(cfg, slide_table, *, command_prefix, indices=None, out_root=None, force=False):
        calls.update(cfg=cfg, table=slide_table, prefix=command_prefix, indices=indices)
        return pd.DataFrame({"status": ["completed", "failed"]})

    _fake_backends(
        monkeypatch,
        config={
            "default_config_path": lambda backend, model: config,
            "load_backend_config": lambda path: {"path": str(path)},
        },
        mussel={"run_table": run_table},
    )
    monkeypatch.setattr(
        envs, "ensure_env", lambda name: pytest.fail("mussel extraction must not dispatch")
    )

    result = runner.invoke(
        cli.app,
        ["extract", str(table), "--backend", "mussel", "--model", "hoptimus0", "--indices", "1-2"],
    )
    assert result.exit_code == 0, result.output
    assert calls["prefix"] == [
        "uv",
        "run",
        "--frozen",
        "--project",
        str(envs.ENVS_DIR / "mussel"),
    ]
    assert calls["cfg"] == {"path": str(config)}
    assert calls["table"] == table
    assert calls["indices"] == [1, 2]


def test_extract_mussel_all_failed_exits_nonzero(monkeypatch, tmp_path):
    table = _slide_table(tmp_path)
    config = tmp_path / "c.toml"
    config.write_text("")
    _fake_backends(
        monkeypatch,
        config={
            "default_config_path": lambda backend, model: config,
            "load_backend_config": lambda path: {},
        },
        mussel={"run_table": lambda cfg, table, **kw: pd.DataFrame({"status": ["failed"] * 2})},
    )
    result = runner.invoke(cli.app, ["extract", str(table), "-b", "mussel", "-m", "hoptimus0"])
    assert result.exit_code == 1


def test_extract_lazyslide_dispatches_first(monkeypatch, tmp_path):
    seen = []

    def fake_ensure(name):
        seen.append(name)
        raise envs.EnvDispatchError("dispatch stopped for test")

    monkeypatch.setattr(envs, "ensure_env", fake_ensure)
    result = runner.invoke(cli.app, ["extract", str(_slide_table(tmp_path))])
    assert seen == ["lazyslide"]
    assert result.exit_code == 1
    assert "dispatch stopped for test" in result.output


@pytest.mark.parametrize(
    "argv",
    [
        ["models"],
        ["aggregate", "uni2"],
        ["qc", "slides.csv"],
        ["visualize"],
        ["run", "slides.csv", "clinical.csv"],
        ["extract-dask", "slides.csv"],
        ["doctor"],
    ],
)
def test_lazyslide_commands_dispatch(monkeypatch, argv):
    def fake_ensure(name):
        raise envs.EnvDispatchError(f"needs {name}")

    monkeypatch.setattr(envs, "ensure_env", fake_ensure)
    result = runner.invoke(cli.app, argv)
    assert result.exit_code == 1
    assert "needs lazyslide" in result.output


def test_experiment_dispatch_only_for_extract_or_aggregate(tmp_path):
    base = '[run]\nname = "t"\nstrategy = "frozen_foundation_trained_head"\n'
    post = tmp_path / "post.toml"
    post.write_text(base)
    assert not cli._experiment_needs_lazyslide(post, None, None)
    broken = tmp_path / "broken.toml"
    broken.write_text("not = [valid")
    assert not cli._experiment_needs_lazyslide(broken, None, None)
    config = Path(__file__).resolve().parents[1] / "configs" / "nigeria-v2.toml"
    if config.is_file():
        assert cli._experiment_needs_lazyslide(config, None, None)


def _lazyslide_config(tmp_path: Path, **params) -> Path:
    body = "\n".join(
        f"{key} = {value!r}".replace("True", "true").replace("False", "false")
        for key, value in params.items()
    )
    path = tmp_path / "lazyslide-hoptimus0.toml"
    path.write_text(f'[backend]\nname = "lazyslide"\nmodel = "hoptimus0"\n[params]\n{body}\n')
    return path


def test_lazyslide_run_config_precedence(tmp_path):
    config = _lazyslide_config(
        tmp_path, model_key="h-optimus-0", tile_px=224, amp=False, num_workers=0
    )
    models, params, cfg = cli._lazyslide_run_config(
        config, None, {"tile_px": None, "amp": None, "batch_size": 8}
    )
    assert models == ["h-optimus-0"]
    assert params == {
        "tile_px": 224,
        "mpp": 0.5,
        "amp": False,
        "num_workers": 0,
        "batch_size": 8,
        "slide_encoder": None,
    }
    assert cfg.backend == "lazyslide" and cfg.model == "hoptimus0"
    assert cfg.params["model_key"] == "h-optimus-0" and cfg.params["batch_size"] == 8
    assert cfg.source == config.resolve()

    _, explicit, _ = cli._lazyslide_run_config(config, None, {"tile_px": 256, "amp": True})
    assert (explicit["tile_px"], explicit["amp"]) == (256, True)

    models, defaults, cfg = cli._lazyslide_run_config(None, None, {})
    assert models == ["uni2"] and defaults["tile_px"] == 256 and defaults["amp"] is True
    assert cfg.source is None

    with pytest.raises(Exception):
        cli._lazyslide_run_config(config, ["uni2"], {})
    mussel_cfg = tmp_path / "m.toml"
    mussel_cfg.write_text('[backend]\nname = "mussel"\nmodel = "hoptimus0"\n')
    with pytest.raises(Exception):
        cli._lazyslide_run_config(mussel_cfg, None, {})


def test_extract_lazyslide_config_and_out_root(monkeypatch, tmp_path):
    monkeypatch.setattr(envs, "ensure_env", lambda name: None)
    table = _slide_table(tmp_path)
    config = _lazyslide_config(tmp_path, model_key="h-optimus-0", tile_px=224, amp=False)
    calls = {}

    def extract_features_batch(**kwargs):
        calls.update(kwargs)
        return pd.DataFrame({"success": [True] * len(kwargs["slide_table"])})

    fake = types.ModuleType("argo_deepmsi.feature_extraction")
    fake.list_available_models = lambda: ["h-optimus-0", "uni2"]
    fake.extract_features_batch = extract_features_batch
    monkeypatch.setitem(sys.modules, "argo_deepmsi.feature_extraction", fake)
    root = tmp_path / "stores" / "lazyslide_224"
    result = runner.invoke(
        cli.app,
        [
            "extract",
            str(table),
            "--config",
            str(config),
            "--out-root",
            str(root),
            "--indices",
            "0-1",
            "--workers",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls["models"] == ["h-optimus-0"]
    assert (calls["tile_px"], calls["amp"], calls["num_workers"]) == (224, False, 2)
    assert calls["out_root"] == root and calls["slide_encoder"] is None
    assert calls["backend_config"].params["tile_px"] == 224
    assert len(calls["slide_table"]) == 2

    calls.clear()
    result = runner.invoke(cli.app, ["extract", str(table), "-m", "uni2", "--max-slides", "1"])
    assert result.exit_code == 0, result.output
    assert (calls["tile_px"], calls["amp"], calls["batch_size"]) == (256, True, 64)
    assert calls["out_root"] is None


def test_extract_rejects_unknown_backend(tmp_path):
    result = runner.invoke(cli.app, ["extract", str(_slide_table(tmp_path)), "-b", "nope"])
    assert result.exit_code != 0


def test_parse_indices():
    assert cli._parse_indices("0-3,7,2") == [0, 1, 2, 3, 7]
    assert cli._parse_indices(None) is None
    with pytest.raises(Exception):
        cli._parse_indices("5-2")


def test_select_slides_spreads_sites(tmp_path):
    df = pd.read_csv(_slide_table(tmp_path))
    chosen = cli._select_slides(df, 3, "sites", seed=0)
    assert set(chosen["SITE"]) == {"UITH", "OAUTHC", "MSK"}
    again = cli._select_slides(df, 3, "sites", seed=0)
    assert list(chosen.index) == list(again.index)
    assert len(cli._select_slides(df, 20, "sites", seed=0)) == len(df)


def test_compare_backends_missing_outputs(monkeypatch, tmp_path):
    table = _slide_table(tmp_path)

    def read_tiles(slide_path, backend, model, *, out_root=None, model_key=None):
        if backend == "mussel" and Path(slide_path).name != "s0.svs":
            raise FileNotFoundError(f"no {model} h5 for {Path(slide_path).name}")
        return object()

    _fake_backends(
        monkeypatch,
        readers={"read_tiles": read_tiles, "LAZYSLIDE_MODEL_KEYS": {}},
        compare={
            "compare_slides": lambda pairs: pytest.fail("must not compare"),
            "verdict": lambda report: "",
            "write_report": lambda *a, **k: None,
        },
    )
    result = runner.invoke(
        cli.app,
        ["compare-backends", "--slides", str(table), "--n", "2", "--out", str(tmp_path / "o")],
    )
    assert result.exit_code == 1
    assert "mussel" in result.output
    assert "--backend mussel" in result.output
    assert "--backend lazyslide" not in result.output
    assert "Missing backend outputs" in result.output


def test_compare_backends_suggests_lazyslide_config(monkeypatch, tmp_path):
    table = _slide_table(tmp_path)

    def read_tiles(slide_path, backend, model, *, out_root=None, model_key=None):
        if backend == "lazyslide":
            raise FileNotFoundError("no store")
        return object()

    _fake_backends(
        monkeypatch,
        readers={"read_tiles": read_tiles, "LAZYSLIDE_MODEL_KEYS": {"hoptimus0": "h-optimus-0"}},
        compare={"compare_slides": None, "verdict": None, "write_report": None},
    )
    result = runner.invoke(
        cli.app, ["compare-backends", "--slides", str(table), "--n", "1", "--out", str(tmp_path)]
    )
    assert result.exit_code == 1
    output = " ".join(result.output.split())
    assert "--config configs/backends/lazyslide-hoptimus0.toml" in output
    assert "--no-amp" not in output


def test_compare_backends_reports_verdict(monkeypatch, tmp_path):
    table = _slide_table(tmp_path)
    captured = {}

    def compare_slides(pairs):
        captured["pairs"] = pairs
        return "report"

    _fake_backends(
        monkeypatch,
        readers={
            "read_tiles": lambda slide_path, backend, model, *, out_root=None, model_key=None: (
                backend,
                model_key or model,
            ),
            "LAZYSLIDE_MODEL_KEYS": {"hoptimus0": "h-optimus-0"},
        },
        compare={
            "compare_slides": compare_slides,
            "verdict": lambda report: "identical up to float noise",
            "write_report": lambda report, out, title, **kw: (
                out / f"{kw['stem']}.csv",
                out / f"{kw['stem']}.md",
            ),
        },
    )
    result = runner.invoke(
        cli.app,
        [
            "compare-backends",
            "--slides",
            str(table),
            "--n",
            "3",
            "--out",
            str(tmp_path / "o"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "identical up to float noise" in result.output
    assert "compare_hoptimus0.csv" in result.output
    assert len(captured["pairs"]) == 3
    _, lazy, mussel = captured["pairs"][0]
    assert lazy == ("lazyslide", "h-optimus-0")
    assert mussel == ("mussel", "hoptimus0")


def test_paladin_without_checkpoint_exits(monkeypatch):
    monkeypatch.delenv("PALADIN_CHECKPOINT", raising=False)
    result = runner.invoke(cli.app, ["paladin"])
    assert result.exit_code == 1
    assert "PALADIN_CHECKPOINT" in result.output


def test_paladin_with_checkpoint_prints_command(monkeypatch, tmp_path):
    project = tmp_path / "paladin"
    (project / ".venv" / "bin").mkdir(parents=True)
    (project / ".venv" / "bin" / "python").write_text("")
    monkeypatch.setitem(
        envs.ENVS, "paladin", envs.EnvSpec("paladin", project, "venv", "lightning", "t")
    )
    monkeypatch.setenv("PALADIN_CHECKPOINT", str(tmp_path / "ckpt.pt"))
    result = runner.invoke(cli.app, ["paladin"])
    assert result.exit_code == 0, result.output
    assert "Would run" in result.output


def test_compare_backends_two_lazyslide_roots_and_slide_embeddings(monkeypatch, tmp_path):
    table = _slide_table(tmp_path)
    seen = []
    captured = {}

    def read_tiles(slide_path, backend, model, *, out_root=None, model_key=None):
        seen.append((backend, out_root, model_key))
        return (backend, out_root)

    def compare_slides(pairs, **kwargs):
        captured["pairs"], captured["kwargs"] = pairs, kwargs
        return "report"

    _fake_backends(
        monkeypatch,
        readers={"read_tiles": read_tiles, "LAZYSLIDE_MODEL_KEYS": {"hoptimus0": "h-optimus-0"}},
        compare={
            "compare_slides": compare_slides,
            "verdict": lambda report: "grid-only divergence",
            "write_report": lambda report, out, title, **kw: (
                out / f"{kw['stem']}.csv",
                out / f"{kw['stem']}.md",
            ),
            "compare_slide_embeddings": lambda a, b: {"cosine": 1.0},
        },
    )
    monkeypatch.setattr(cli, "_slide_embedding", lambda *args: [1.0, 2.0])
    out = tmp_path / "o"
    out.mkdir()
    result = runner.invoke(
        cli.app,
        [
            "compare-backends",
            "--slides",
            str(table),
            "--n",
            "2",
            "--a",
            "lazyslide=stores/lazyslide_224",
            "--b",
            "lazyslide=stores/lazyslide_224_amp",
            "--anchor",
            "center",
            "--slide-embedding",
            "--stem",
            "compare_fp16",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "compare_fp16.csv" in result.output
    assert captured["kwargs"] == {"anchor": "center"}
    assert {root for _, root, _ in seen} == {
        Path("stores/lazyslide_224"),
        Path("stores/lazyslide_224_amp"),
    }
    assert all(key == "h-optimus-0" for _, _, key in seen)
    assert len(pd.read_csv(out / "compare_fp16_slide.csv")) == 2

    same = runner.invoke(
        cli.app, ["compare-backends", "--slides", str(table), "--a", "mussel", "--b", "mussel"]
    )
    assert same.exit_code != 0
