"""Synthetic tests for extraction backends: configs, Mussel runner, readers, comparison."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from argo_deepmsi.backends import compare, mussel, readers
from argo_deepmsi.backends.config import (
    BackendConfig,
    default_config_path,
    load_backend_config,
)
from argo_deepmsi.backends.provenance import (
    collect_provenance,
    read_provenance,
    write_provenance,
)

h5py = pytest.importorskip("h5py")

REPO_CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "backends"
MUSSEL_CFG = BackendConfig("mussel", "hoptimus0", {"model_type": "OPTIMUS"})


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _grid(n: int, edge: float, origin=(0, 0)) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(n), np.arange(n))
    coords = np.stack([xs.ravel(), ys.ravel()], axis=1) * edge
    return (coords + np.asarray(origin)).astype(np.int64)


def _tiles(coords, features, edge=512.0, backend="lazyslide") -> readers.TileFeatures:
    return readers.TileFeatures(
        features=features,
        coords=coords,
        tile_px_level0=edge,
        tile_px=256,
        mpp=0.5,
        backend=backend,
        model="m",
        slide_id="s",
    )


# --- config ------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(p.name for p in REPO_CONFIGS.glob("*.toml")))
def test_checked_in_configs_load(name):
    cfg = load_backend_config(REPO_CONFIGS / name)
    backend, stem_model = name.removesuffix(".toml").split("-", 1)
    assert cfg.backend == backend
    # Variants (e.g. lazyslide-hoptimus0-224.toml) name the model plus a suffix.
    assert stem_model == cfg.model or stem_model.startswith(f"{cfg.model}-")
    if stem_model == cfg.model:
        assert default_config_path(backend, cfg.model).resolve() == (REPO_CONFIGS / name).resolve()


def test_lazyslide_configs_encode_study_settings():
    variant = load_backend_config(REPO_CONFIGS / "lazyslide-hoptimus0-224.toml").params
    assert (variant["model_key"], variant["tile_px"], variant["mpp"]) == ("h-optimus-0", 224, 0.5)
    assert variant["amp"] is False
    titan = load_backend_config(REPO_CONFIGS / "lazyslide-titan.toml").params
    assert (titan["model_key"], titan["slide_encoder"]) == ("conch_v1.5", "titan")
    assert (titan["tile_px"], titan["mpp"]) == (512, 0.5)


def test_config_validation(tmp_path):
    good = _write(
        tmp_path / "ok.toml",
        '[backend]\nname = "mussel"\nmodel = "hoptimus0"\n[params]\nmodel_type = "OPTIMUS"\n',
    )
    cfg = load_backend_config(good)
    assert cfg.params == {"model_type": "OPTIMUS"} and cfg.source == good.resolve()

    bad_cases = {
        "backend.toml": '[backend]\nname = "timm"\nmodel = "x"\n',
        "top.toml": '[backend]\nname = "mussel"\nmodel = "x"\n[extra]\na = 1\n',
        "key.toml": '[backend]\nname = "mussel"\nmodel = "x"\nversion = 1\n',
        "missing.toml": '[backend]\nname = "mussel"\n',
        "lazy.toml": '[backend]\nname = "lazyslide"\nmodel = "x"\n[params]\ntile_size = 1\n',
        "reserved.toml": '[backend]\nname = "mussel"\nmodel = "x"\n[params]\nslide_path = "a"\n',
    }
    for name, text in bad_cases.items():
        with pytest.raises(ValueError):
            load_backend_config(_write(tmp_path / name, text))


# --- mussel argv + runner --------------------------------------------------------------


def test_build_argv_only_explicit_params_and_deterministic(tmp_path):
    outputs = mussel.output_paths(tmp_path / "slides" / "a.svs", "hoptimus0")
    assert outputs.dir == tmp_path / "slides" / "a.mussel"
    assert outputs.h5.name == "hoptimus0.features.h5"
    argv = mussel.build_argv(MUSSEL_CFG, tmp_path / "slides" / "a.svs", outputs)
    assert argv[0] == "tessellate_extract_features"
    assert [a.split("=", 1)[0] for a in argv[1:]] == [
        "slide_path",
        "output_h5_path",
        "output_pt_path",
        "model_type",
    ]
    assert "model_type=OPTIMUS" in argv

    params_a = {"seg_config": {"mpp": 0.5}, "batch_size": 8, "model_type": "OPTIMUS"}
    params_b = {"model_type": "OPTIMUS", "batch_size": 8, "seg_config": {"mpp": 0.5}}
    argv_a = mussel.build_argv(BackendConfig("mussel", "m", params_a), "a.svs", outputs)
    argv_b = mussel.build_argv(BackendConfig("mussel", "m", params_b), "a.svs", outputs)
    assert argv_a == argv_b
    assert argv_a[4:] == ["model_type=OPTIMUS", "batch_size=8", "seg_config.mpp=0.5"]

    titan = mussel.build_argv(BackendConfig("mussel", "titan", {}), "a.svs", outputs)
    assert titan[4:6] == ["model_type=CONCH1_5", "slide_model_type=TITAN_SLIDE"]
    with pytest.raises(ValueError):
        mussel.build_argv(BackendConfig("mussel", "unknown", {}), "a.svs", outputs)

    mirrored = mussel.output_paths("/data/site/a.svs", "hoptimus0", out_root=tmp_path)
    assert mirrored.dir == tmp_path / "data" / "site" / "a.mussel"


def _arg(argv, key):
    return next(a.split("=", 1)[1] for a in argv if a.startswith(f"{key}="))


class FakeMussel:
    """Stand-in for ``subprocess.run`` that writes Mussel-shaped outputs."""

    def __init__(self, returncode=0, n=4, write=True):
        self.returncode, self.n, self.write, self.calls = returncode, n, write, []

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        if "-c" in argv:
            return subprocess.CompletedProcess(argv, 0, stdout="9.9.9\n", stderr="")
        tmpdir = Path(kwargs["env"]["TMPDIR"])
        # short per-run TMPDIR (unix sockets cap paths at 108 bytes), never /tmp
        assert tmpdir.is_dir() and tmpdir.parent.name == ".tmp_build"
        assert len(str(tmpdir)) < 90 and tmpdir.name.startswith("mussel-")
        self.tmpdirs = getattr(self, "tmpdirs", []) + [tmpdir]
        if self.write:
            with h5py.File(_arg(argv, "output_h5_path"), "w") as handle:
                handle.create_dataset("coords", data=_grid(2, 448)[: self.n])
                handle.create_dataset("features", data=np.ones((self.n, 3), np.float32))
            Path(_arg(argv, "output_pt_path")).write_bytes(b"pt")
        return subprocess.CompletedProcess(argv, self.returncode, stdout="log", stderr="boom")


def test_run_slide_success_skip_and_drift(tmp_path):
    slide = _write(tmp_path / "a.svs", "slide")
    runner = FakeMussel()
    prefix = ["uv", "run", "--project", "envs/mussel"]
    row = mussel.run_slide(MUSSEL_CFG, slide, command_prefix=prefix, runner=runner)
    outputs = mussel.output_paths(slide, "hoptimus0")
    assert row["status"] == "ok" and mussel.is_complete(outputs)
    assert runner.calls[0][: len(prefix)] == prefix
    assert not list(outputs.dir.glob(".partial-*"))
    assert not any(path.exists() for path in runner.tmpdirs)
    prov = read_provenance(outputs.provenance)
    assert prov["backend"] == "mussel" and prov["params"] == {"model_type": "OPTIMUS"}
    assert prov["extra"]["n_rows"] == 4 and prov["tool"]["commit"] == mussel.MUSSEL_PINNED_COMMIT

    again = mussel.run_slide(MUSSEL_CFG, slide, command_prefix=prefix, runner=runner)
    assert again["status"] == "skipped" and len(runner.calls) == 1

    drifted = BackendConfig("mussel", "hoptimus0", {"model_type": "OPTIMUS", "batch_size": 8})
    drift = mussel.run_slide(drifted, slide, command_prefix=prefix, runner=runner)
    assert drift["status"] == "failed" and "different params" in drift["error"]
    forced = mussel.run_slide(drifted, slide, command_prefix=prefix, runner=runner, force=True)
    assert forced["status"] == "ok" and len(runner.calls) == 2


@pytest.mark.parametrize(
    "runner",
    [FakeMussel(returncode=1), FakeMussel(write=False), FakeMussel(n=0)],
    ids=["nonzero-exit", "no-output", "empty-h5"],
)
def test_run_slide_failure_is_isolated(tmp_path, runner):
    slide = _write(tmp_path / "a.svs", "slide")
    row = mussel.run_slide(MUSSEL_CFG, slide, command_prefix=[], runner=runner)
    outputs = mussel.output_paths(slide, "hoptimus0")
    assert row["status"] == "failed" and row["error"]
    assert not outputs.h5.exists() and not outputs.provenance.exists()
    assert not list(outputs.dir.glob(".partial-*"))
    assert outputs.failed_log.is_file()


def test_run_slide_runner_exception_and_missing_slide(tmp_path):
    def explode(argv, **kwargs):
        raise RuntimeError("CUDA gone")

    slide = _write(tmp_path / "a.svs", "slide")
    row = mussel.run_slide(MUSSEL_CFG, slide, command_prefix=[], runner=explode)
    assert row["status"] == "failed" and "CUDA gone" in row["error"]
    missing = mussel.run_slide(MUSSEL_CFG, tmp_path / "nope.svs", command_prefix=[])
    assert missing["status"] == "failed" and "not found" in missing["error"]


def test_run_slide_titan_multi_model_layout(tmp_path):
    def titan_runner(argv, **kwargs):
        root = Path(_arg(argv, "output_h5_path")).parent
        for model_type, feats in (("TITAN_SLIDE", np.ones((1, 768))), ("CONCH1_5", None)):
            for sub in ("h5", "pt"):
                (root / model_type / sub).mkdir(parents=True)
            with h5py.File(root / model_type / "h5" / "a.features.h5", "w") as handle:
                if feats is None:
                    handle.create_dataset("coords", data=_grid(2, 985))
                    handle.create_dataset("features", data=np.ones((4, 768), np.float32))
                else:
                    handle.create_dataset("features", data=feats.astype(np.float32))
            (root / model_type / "pt" / "a.features.pt").write_bytes(b"pt")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    slide = _write(tmp_path / "a.svs", "slide")
    cfg = BackendConfig(
        "mussel", "titan", {"model_type": "CONCH1_5", "slide_model_type": "TITAN_SLIDE"}
    )
    row = mussel.run_slide(cfg, slide, command_prefix=[], runner=titan_runner)
    outputs = mussel.output_paths(slide, "titan")
    assert row["status"] == "ok" and mussel.is_complete(outputs)
    assert readers.read_mussel_slide_embedding(outputs.h5).shape == (768,)
    tiles = readers.read_tiles(slide, "mussel", "titan")
    assert tiles.n_tiles == 4 and tiles.model == "titan" and tiles.tile_px_level0 == 985


def test_run_table_indices(tmp_path):
    slides = [_write(tmp_path / f"s{i}.svs", "x") for i in range(3)]
    table = tmp_path / "slides.csv"
    pd.DataFrame({"FILENAME": [str(p) for p in slides]}).to_csv(table, index=False)
    runner = FakeMussel()
    df = mussel.run_table(MUSSEL_CFG, table, command_prefix=[], indices=[1, 2, 7], runner=runner)
    assert df["row"].tolist() == [1, 2] and (df["status"] == "ok").all()
    prov = read_provenance(mussel.output_paths(slides[1], "hoptimus0").provenance)
    assert prov["tool"]["version"] == "9.9.9"
    assert not mussel.output_paths(slides[0], "hoptimus0").dir.exists()


# --- readers -----------------------------------------------------------------------------


def _write_mussel_h5(path: Path, coords, features=None, attrs=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        dset = handle.create_dataset("coords", data=coords)
        for key, value in (attrs or {}).items():
            dset.attrs[key] = value
        if features is not None:
            handle.create_dataset("features", data=features)
    return path


def test_read_mussel_float32(tmp_path):
    coords = _grid(3, 448)
    feats = np.random.default_rng(0).normal(size=(9, 5)).astype(np.float32)
    h5 = _write_mussel_h5(
        tmp_path / "a.mussel" / "hoptimus0.features.h5",
        coords,
        feats,
        {"patch_size": 448, "mpp": 0.5, "native_mpp": 0.25},
    )
    tiles = readers.read_mussel(h5)
    np.testing.assert_array_equal(tiles.features, feats)
    np.testing.assert_array_equal(tiles.coords, coords)
    assert (tiles.tile_px_level0, tiles.tile_px, tiles.mpp) == (448.0, 224, 0.5)
    assert (tiles.slide_id, tiles.model, tiles.backend) == ("a", "hoptimus0", "mussel")


def test_read_mussel_bfloat16_void_without_attrs(tmp_path):
    values = np.array([[1.0, -2.5, 0.15625], [3.0, 0.0, -0.75]], dtype=np.float32)
    bf16 = (values.view(np.uint32) >> 16).astype(np.uint16).view("V2")
    h5 = _write_mussel_h5(tmp_path / "hoptimus0.features.h5", _grid(2, 512)[:2], bf16)
    tiles = readers.read_mussel(h5)
    np.testing.assert_array_equal(tiles.features, values)
    assert tiles.tile_px_level0 == 512.0 and tiles.mpp is None


def test_read_mussel_pt_fallback_and_slide_embedding(tmp_path):
    torch = pytest.importorskip("torch")
    feats = np.arange(8, dtype=np.float32).reshape(4, 2)
    h5 = _write_mussel_h5(tmp_path / "x.mussel" / "hoptimus0.features.h5", _grid(2, 448))
    torch.save(torch.from_numpy(feats).to(torch.bfloat16), h5.with_name("hoptimus0.features.pt"))
    np.testing.assert_array_equal(readers.read_mussel(h5).features, feats)

    slide_h5 = tmp_path / "titan.features.h5"
    with h5py.File(slide_h5, "w") as handle:
        handle.create_dataset("features", data=np.ones((1, 6), np.float32))
    assert readers.read_mussel_slide_embedding(slide_h5).shape == (6,)
    with pytest.raises(KeyError):
        readers.read_mussel(slide_h5)


def _fake_lazyslide_store(zarr_path: Path, key: str, coords, feats, edge=492, shuffle=True):
    ad = pytest.importorskip("anndata")
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    shapely = pytest.importorskip("shapely")

    n = len(coords)
    tile_ids = np.arange(n)
    obs = pd.DataFrame({"tile_id": tile_ids, "library_id": "tiles"}, index=tile_ids.astype(str))
    ad.AnnData(X=feats, obs=obs).write_zarr(str(zarr_path / "tables" / f"{key}_tiles"))
    (zarr_path / "tables" / f"{key}_tiles" / ".zattrs").write_text(
        json.dumps({"region": "tiles", "instance_key": "tile_id"})
    )
    order = np.random.default_rng(1).permutation(n) if shuffle else np.arange(n)
    polys = [shapely.box(x, y, x + edge, y + edge) for x, y in coords[order]]
    shapes_dir = zarr_path / "shapes" / "tiles"
    shapes_dir.mkdir(parents=True)
    table = pa.table(
        {
            "tile_id": pa.array(tile_ids[order], pa.int64()),
            "tissue_id": pa.array(np.zeros(n, np.int64)),
            "geometry": pa.array(shapely.to_wkb(polys).tolist(), pa.binary()),
        }
    )
    pq.write_table(table, shapes_dir / "shapes.parquet")
    spec = {"tiles": {"width": 256, "height": 256, "mpp": 0.5, "base_downsample": edge / 256}}
    (zarr_path / "zarr.json").write_text(
        json.dumps({"attributes": {"tile_spec": spec, "slide_properties": {"mpp": 0.26}}})
    )


def test_read_lazyslide_fake_store(tmp_path):
    slide = tmp_path / "site" / "b.svs"
    coords = _grid(3, 492, origin=(100, 200))
    feats = np.random.default_rng(2).normal(size=(9, 4)).astype(np.float32)
    _fake_lazyslide_store(tmp_path / "site" / "b.zarr", "h-optimus-0", coords, feats)
    tiles = readers.read_tiles(slide, "lazyslide", "hoptimus0")
    np.testing.assert_array_equal(tiles.coords, coords)
    np.testing.assert_array_equal(tiles.features, feats)
    assert tiles.tile_px == 256 and tiles.tile_px_level0 == pytest.approx(492.0)
    assert (tiles.slide_id, tiles.model, tiles.mpp) == ("b", "h-optimus-0", 0.5)
    assert tiles.provenance["slide_mpp"] == 0.26


def test_lazyslide_store_out_root_and_provenance(tmp_path):
    slide = tmp_path / "data" / "site" / "b.pyramidal.tiff"
    assert readers.lazyslide_store(slide) == slide.parent / "b.pyramidal.zarr"
    root = tmp_path / "stores" / "lazyslide_224"
    store = readers.lazyslide_store(slide, root)
    assert store.name == "b.pyramidal.zarr" and store.is_relative_to(root)
    assert str(store).endswith(str(slide.parent.relative_to(slide.anchor) / "b.pyramidal.zarr"))

    coords = _grid(2, 431)
    feats = np.ones((4, 3), np.float32)
    _fake_lazyslide_store(store, "h-optimus-0", coords, feats, edge=431)
    prov_path = readers.lazyslide_provenance_path(store, "h-optimus-0")
    assert prov_path == store.parent / "b.pyramidal.zarr.h-optimus-0.provenance.json"
    prov_path.write_text(json.dumps({"backend": "lazyslide", "params": {"tile_px": 224}}))
    tiles = readers.read_tiles(slide, "lazyslide", "hoptimus0", out_root=root)
    assert tiles.provenance["provenance"]["params"] == {"tile_px": 224}
    assert tiles.n_tiles == 4


# --- comparison ---------------------------------------------------------------------------


def test_match_tiles_center_anchor_for_different_tile_sizes():
    feats = np.eye(4, dtype=np.float32)
    small = _tiles(np.array([[0, 0], [1000, 0], [0, 1000], [1000, 1000]]), feats, edge=431)
    # 492 px tiles whose centres coincide with the 431 px tiles' centres
    big = _tiles(small.coords - 30, feats, edge=492)
    assert len(compare.match_tiles(small, big, tol_frac=0.05)[0]) == 0
    ia, ib = compare.match_tiles(small, big, tol_frac=0.05, anchor="center")
    assert np.array_equal(ia, np.arange(4)) and np.array_equal(ib, np.arange(4))
    df = compare.compare_slides([("s", small, big)], anchor="center")
    assert df.loc[0, "n_matched"] == 4
    with pytest.raises(ValueError):
        compare.match_tiles(small, big, anchor="corner")


def test_match_tiles_shift_tolerance_and_fallback(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    coords = _grid(5, 512)
    a = _tiles(coords, rng.normal(size=(25, 8)))
    within = _tiles(coords + [100, -60], a.features)
    beyond = _tiles(coords + [200, 0], a.features)
    ia, ib = compare.match_tiles(a, within)
    assert len(ia) == 25 and np.array_equal(ia, ib)
    assert len(compare.match_tiles(a, beyond)[0]) == 0
    assert compare.compare_pair(a, beyond)["grid_dx_median"] == 200

    subset = _tiles(coords[::2] + 30, a.features[::2])
    ia, ib = compare.match_tiles(a, subset)
    assert np.array_equal(ia, np.arange(0, 25, 2)) and np.array_equal(ib, np.arange(13))

    import builtins

    real_import = builtins.__import__

    def no_scipy(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_scipy)
    ia2, ib2 = compare.match_tiles(a, subset)
    assert np.array_equal(ia, ia2) and np.array_equal(ib, ib2)


def test_compare_pair_and_verdicts(tmp_path):
    rng = np.random.default_rng(4)
    coords = _grid(6, 512)
    feats = rng.normal(size=(36, 16)).astype(np.float32)
    a = _tiles(coords, feats)
    same = _tiles(coords + 3, feats + 1e-7, backend="mussel")
    noisy = _tiles(coords, feats + rng.normal(scale=0.5, size=feats.shape), backend="mussel")
    partial = _tiles(coords[:18], feats[:18], backend="mussel")

    identical = compare.compare_pair(a, same)
    assert identical["cos_median"] == pytest.approx(1.0) and identical["n_matched"] == 36
    assert identical["grid_dx_median"] == 3 and identical["max_abs_diff"] < 1e-5
    assert identical["slide_mean_pearson_all"] == pytest.approx(1.0)
    assert identical["n_exact"] == 0 and np.isnan(identical["cos_median_exact"])
    assert identical["coverage_iou"] > 0.98
    assert compare.coverage_iou(a, partial) == pytest.approx(0.5)
    exact = compare.compare_pair(a, _tiles(coords, feats, backend="mussel"))
    assert exact["n_exact"] == 36 and exact["cos_median_exact"] == pytest.approx(1.0)
    perturbed = compare.compare_pair(a, noisy)
    assert perturbed["cos_median"] < 0.99 and perturbed["rel_l2"] > 0.1

    df_same = compare.compare_slides([("s1", a, same)])
    assert compare.verdict(df_same).startswith(compare.IDENTICAL)
    df_grid = compare.compare_slides([("s1", a, same), ("s2", a, partial)])
    assert compare.verdict(df_grid).startswith(compare.GRID_ONLY)
    df_loc = compare.compare_slides([("s1", a, same), ("s2", a, noisy)])
    assert compare.verdict(df_loc).startswith(compare.LOCATED) and "s2" in compare.verdict(df_loc)
    wrong_dim = _tiles(coords, feats[:, :8])
    assert "dimensions" in compare.verdict(compare.compare_slides([("s3", a, wrong_dim)]))

    csv_path, md_path = compare.write_report(df_grid, tmp_path / "report", "Compare hoptimus0")
    assert csv_path.name == "compare_hoptimus0.csv" and len(pd.read_csv(csv_path)) == 2
    assert compare.GRID_ONLY in md_path.read_text()

    emb = rng.normal(size=768)
    assert compare.compare_slide_embeddings(emb, emb)["cosine"] == pytest.approx(1.0)
    assert not compare.compare_slide_embeddings(emb, emb[:10])["dim_match"]


# --- provenance ---------------------------------------------------------------------------


def test_provenance_round_trip(tmp_path):
    lock = _write(tmp_path / "uv.lock", "lock")
    cfg_path = _write(
        tmp_path / "c.toml", '[backend]\nname = "mussel"\nmodel = "titan"\n[params]\nx = 1\n'
    )
    cfg = load_backend_config(cfg_path)
    prov = collect_provenance(
        cfg, tool_version="1.0", tool_commit="abc", env_lock=lock, extra={"path": tmp_path}
    )
    assert prov["env_lock"]["sha256"] and prov["config_sha256"]
    assert prov["tool"] == {"version": "1.0", "commit": "abc"}
    assert {"argo_git", "created_at", "hostname", "python"} <= set(prov)
    write_provenance(tmp_path / "p.json", prov)
    assert read_provenance(tmp_path / "p.json") == prov
