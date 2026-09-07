"""End-to-end tests for the argo_deepmsi pipeline.

These tests exercise the *argo_deepmsi* functions directly, not LazySlide's
API. They validate our wiring: feature_extraction, aggregation (including the
slide-outer/model-inner refactor), AnnData output, QC filter,
StratifiedGroupKFold plumbing, and the cached-zarr visualization helper.

CPU-only: uses resnet50 on LazySlide's public GTEx sample slide (~253 tiles).

Run:
    pytest tests/test_argo_pipeline.py -v --tb=short
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.network


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def workspace_slide(sample_slide, data_dir) -> Path:
    """Copy the GTEx sample slide into a clean dir we own, so ``wsi.write()``
    can place the zarr next to it without polluting the HF cache."""
    ws = data_dir / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    dest = ws / sample_slide.name
    if not dest.exists():
        shutil.copy2(sample_slide, dest)
    return dest


@pytest.fixture(scope="session")
def extracted_zarr(workspace_slide) -> Path:
    """Run argo's extract_features_single_slide once on CPU with resnet50.

    Returns the zarr path. Session-scoped so later tests reuse it.
    """
    from argo_deepmsi.feature_extraction import extract_features_single_slide

    zarr_path = extract_features_single_slide(
        slide_path=workspace_slide,
        models=["resnet50"],
        tile_px=256,
        mpp=0.5,
        amp=False,
        device="cpu",
        num_workers=0,
        batch_size=16,
    )
    assert zarr_path is not None and Path(zarr_path).exists()
    return Path(zarr_path)


@pytest.fixture(scope="session")
def slide_table_csv(workspace_slide, extracted_zarr, data_dir) -> Path:
    """Three-row slide table backed by the same real zarr (simulates 3
    patients, each with one slide). Sufficient rows to run
    StratifiedGroupKFold(n_splits=2)."""
    # Create symlink "slides" so multiple rows resolve independently.
    # NB: symlink_to records the literal target string — pass resolved paths
    # so the link works regardless of the kernel's CWD when resolving it.
    ws = workspace_slide.parent
    slide_target = workspace_slide.resolve()
    zarr_target = extracted_zarr.resolve()
    rows = []
    for pid, _ in [("P001", "MSI-H"), ("P002", "MSS"), ("P003", "MSI-H")]:
        link = ws / f"{pid}_{workspace_slide.name}"
        if not link.exists():
            link.symlink_to(slide_target)
        zarr_link = ws / f"{pid}_{workspace_slide.stem}.zarr"
        if not zarr_link.exists():
            zarr_link.symlink_to(zarr_target)
        rows.append(
            {"PATIENT": pid, "FILENAME": str(link.resolve().parent / link.name), "SITE": "TEST"}
        )
    df = pd.DataFrame(rows)
    path = data_dir / "slide_table.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def clinical_table_csv(data_dir) -> Path:
    df = pd.DataFrame(
        {
            "PATIENT": ["P001", "P002", "P003"],
            "isMSIH": ["MSI-H", "MSS", "MSI-H"],
        }
    )
    path = data_dir / "clinical_table.csv"
    df.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


class TestExtractFeaturesSingleSlide:
    def test_creates_zarr_with_feature_table(self, extracted_zarr):
        assert extracted_zarr.exists()
        assert (extracted_zarr / "tables" / "resnet50_tiles").exists()

    def test_incremental_skips_existing_model(self, workspace_slide, extracted_zarr):
        """Re-running with the same model should be a fast no-op: still
        returns the zarr, doesn't re-extract."""
        from argo_deepmsi.feature_extraction import extract_features_single_slide

        out = extract_features_single_slide(
            slide_path=workspace_slide,
            models=["resnet50"],
            device="cpu",
            num_workers=0,
            batch_size=16,
        )
        assert out == extracted_zarr


# --------------------------------------------------------------------------
# Aggregation (simple pooling — the slide-outer/model-inner refactor)
# --------------------------------------------------------------------------


class TestAggregateSimplePooling:
    def test_mean_pooling_writes_all_outputs(self, slide_table_csv, tmp_path):
        from argo_deepmsi.feature_extraction import aggregate_simple_pooling

        out = tmp_path / "agg"
        results = aggregate_simple_pooling(
            slide_table=slide_table_csv,
            models=["resnet50"],
            method="mean",
            output_dir=out,
        )
        assert "resnet50" in results
        df = results["resnet50"]
        assert len(df) == 3  # three simulated slides

        # All three output formats are present
        assert (out / "embeddings.npy").exists()
        assert (out / "metadata.csv").exists()
        assert (out / "embeddings.h5ad").exists()

        # Shapes line up
        X = np.load(out / "embeddings.npy")
        assert X.shape[0] == 3
        assert X.shape[1] > 0
        meta = pd.read_csv(out / "metadata.csv")
        assert list(meta["patient_id"]) == ["P001", "P002", "P003"]

    def test_h5ad_contains_obs_and_X(self, slide_table_csv, tmp_path):
        import anndata as ad
        from argo_deepmsi.feature_extraction import aggregate_simple_pooling

        out = tmp_path / "agg_h5ad"
        aggregate_simple_pooling(
            slide_table=slide_table_csv,
            models=["resnet50"],
            method="mean",
            output_dir=out,
        )
        adata = ad.read_h5ad(out / "embeddings.h5ad")
        assert adata.n_obs == 3
        assert {"patient_id", "site", "zarr_path"}.issubset(adata.obs.columns)
        assert adata.uns.get("model") == "resnet50"
        assert adata.uns.get("aggregation") == "mean"

    def test_rejects_unknown_method(self, slide_table_csv, tmp_path):
        from argo_deepmsi.feature_extraction import aggregate_simple_pooling

        with pytest.raises(ValueError):
            aggregate_simple_pooling(
                slide_table=slide_table_csv,
                models=["resnet50"],
                method="not-a-method",  # type: ignore[arg-type]
                output_dir=tmp_path,
            )

    def test_mean_matches_direct_numpy(self, slide_table_csv, tmp_path):
        """Sanity: aggregate_simple_pooling('mean') == AnnData.X.mean(axis=0)."""
        import anndata as ad
        import zarr as _zarr
        from argo_deepmsi.feature_extraction import aggregate_simple_pooling

        out = tmp_path / "agg_sanity"
        aggregate_simple_pooling(
            slide_table=slide_table_csv,
            models=["resnet50"],
            method="mean",
            output_dir=out,
        )
        agg = ad.read_h5ad(out / "embeddings.h5ad")

        # Pull the X matrix directly from one of the zarrs and mean it
        zarr_path = Path(agg.obs.iloc[0]["zarr_path"])
        store = _zarr.open(str(zarr_path), mode="r")
        X = np.asarray(store["tables"]["resnet50_tiles"]["X"][:])
        expected = X.mean(axis=0)

        np.testing.assert_allclose(agg.X[0], expected.astype(np.float32), rtol=1e-4)


# --------------------------------------------------------------------------
# Training data loading + GroupKFold plumbing
# --------------------------------------------------------------------------


class TestTrainingWiring:
    def test_load_training_data_prefers_h5ad(self, slide_table_csv, clinical_table_csv, tmp_path):
        from argo_deepmsi.feature_extraction import aggregate_simple_pooling
        from argo_deepmsi.training import load_training_data

        emb_dir = tmp_path / "agg_for_training"
        aggregate_simple_pooling(
            slide_table=slide_table_csv,
            models=["resnet50"],
            method="mean",
            output_dir=emb_dir,
        )
        # Sanity: both npy and h5ad exist; load_training_data should pick h5ad
        assert (emb_dir / "embeddings.h5ad").exists()
        assert (emb_dir / "embeddings.npy").exists()

        X, y, merged = load_training_data(
            embeddings_dir=emb_dir,
            clinical_table=clinical_table_csv,
        )
        assert X.shape[0] == 3
        assert y.tolist() == [1, 0, 1]
        assert list(merged["patient_id"]) == ["P001", "P002", "P003"]

    def test_load_training_data_row_count_guard(self, tmp_path, clinical_table_csv):
        """If metadata.csv and embeddings.npy disagree on row count, error out
        instead of silently mis-aligning."""
        from argo_deepmsi.training import load_training_data

        d = tmp_path / "bad"
        d.mkdir()
        np.save(d / "embeddings.npy", np.zeros((3, 8), dtype=np.float32))
        pd.DataFrame(
            {
                "slide_id": ["a", "b"],  # 2 rows, not 3
                "patient_id": ["P001", "P002"],
                "site": ["X", "X"],
                "n_tiles": [10, 10],
                "zarr_path": ["x", "y"],
            }
        ).to_csv(d / "metadata.csv", index=False)

        with pytest.raises(ValueError, match="Row count mismatch"):
            load_training_data(embeddings_dir=d, clinical_table=clinical_table_csv)

    def test_load_training_data_preserves_embedding_rows(self, tmp_path):
        """An unmatched metadata row must not shift the matched embeddings."""
        from argo_deepmsi.training import load_training_data

        embedding_dir = tmp_path / "partially_matched"
        embedding_dir.mkdir()
        embeddings = np.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
        np.save(embedding_dir / "embeddings.npy", embeddings)
        pd.DataFrame(
            {
                "slide_id": ["unmatched", "matched-2", "matched-1"],
                "patient_id": ["PX", "P2", "P1"],
            }
        ).to_csv(embedding_dir / "metadata.csv", index=False)
        clinical = tmp_path / "clinical.csv"
        pd.DataFrame(
            {
                "PATIENT": ["P1", "P2"],
                "isMSIH": ["MSI-H", "MSS"],
            }
        ).to_csv(clinical, index=False)

        X, y, merged = load_training_data(embedding_dir, clinical)

        np.testing.assert_array_equal(X, embeddings[[1, 2]])
        assert y.tolist() == [0, 1]
        assert merged["patient_id"].tolist() == ["P2", "P1"]
        assert "_embedding_row" not in merged

    def test_compare_classifiers_requires_groups(self):
        from argo_deepmsi.training import compare_classifiers

        X = np.random.RandomState(0).randn(20, 8).astype(np.float32)
        y = np.array([0, 1] * 10)
        with pytest.raises(ValueError, match="groups"):
            compare_classifiers(X, y, n_splits=2)

    def test_compare_classifiers_group_kfold_runs(self):
        """With valid groups (>=n_splits distinct patients per class),
        compare_classifiers returns one row per classifier with a numeric
        AUROC, proving StratifiedGroupKFold + cross_val_score are plumbed
        correctly."""
        from argo_deepmsi.training import compare_classifiers

        rng = np.random.RandomState(0)
        # 10 patients, 2 slides each = 20 rows; keep label constant within patient
        patient_ids = np.repeat([f"P{i:02d}" for i in range(10)], 2)
        labels = np.repeat([0, 1] * 5, 2)
        # Make the signal linearly separable so the classifiers don't NaN out
        X = rng.randn(20, 8).astype(np.float32) + labels.reshape(-1, 1) * 3.0

        results = compare_classifiers(X, labels, groups=patient_ids, n_splits=3, random_state=0)
        assert len(results) == 3  # LR, RF, SVM
        assert results["auroc_mean"].notna().all()

    def test_scaled_classifier_exposes_fitted_pipeline(self):
        """The reusable estimator applies the same scaling used in training."""
        from argo_deepmsi.training import train_logistic_regression

        rng = np.random.RandomState(1)
        groups = np.repeat([f"P{i:02d}" for i in range(8)], 2)
        y = np.repeat([0, 1] * 4, 2)
        X = rng.randn(16, 4) + y[:, None]

        result = train_logistic_regression(X, y, groups=groups, n_splits=2)

        np.testing.assert_array_equal(
            result["pipeline"].predict(X),
            result["model"].predict(result["scaler"].transform(X)),
        )


# --------------------------------------------------------------------------
# QC filter
# --------------------------------------------------------------------------


class TestQCFilter:
    def test_filter_slides_by_qc_mean_reduce(self, slide_table_csv, tmp_path):
        """Uses resnet50 as a stand-in QC model (helper is model-agnostic:
        any table named {qc_model}_tiles with a 2D X works)."""
        from argo_deepmsi.feature_extraction import filter_slides_by_qc

        out = tmp_path / "qc.csv"
        # All rows back the same zarr, so every score is identical. Setting
        # threshold = that score should keep all 3; thr < it should drop all.
        df_keep = filter_slides_by_qc(
            slide_table=slide_table_csv,
            qc_model="resnet50",
            threshold=1e18,
            reduce="mean",
            output_csv=out,
        )
        assert int(df_keep["passes_qc"].sum()) == 3
        assert out.exists()

        df_drop = filter_slides_by_qc(
            slide_table=slide_table_csv,
            qc_model="resnet50",
            threshold=-1e18,
            reduce="mean",
        )
        assert int(df_drop["passes_qc"].sum()) == 0
        # qc_score column populated with finite numbers
        assert df_drop["qc_score"].notna().all()


# --------------------------------------------------------------------------
# Cached-zarr visualization helper
# --------------------------------------------------------------------------


class TestOpenCached:
    def test_reuses_zarr_without_extraction(self, workspace_slide, extracted_zarr):
        """_open_cached should hit the existing zarr (no GPU re-extraction)
        and expose the pre-computed feature table."""
        from argo_deepmsi.visualization import _open_cached

        wsi = _open_cached(workspace_slide, tile_px=256, mpp=0.5, model="resnet50")
        assert "resnet50_tiles" in wsi.tables
        assert wsi.tables["resnet50_tiles"].n_obs > 0
