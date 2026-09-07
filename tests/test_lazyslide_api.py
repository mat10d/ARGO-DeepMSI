"""
ARGO-DeepMSI: LazySlide API Validation Tests
=============================================

Tests that verify the LazySlide API patterns described in docs/lazyslide_reference_guide.md
actually work. Uses LazySlide's public GTEx data from HuggingFace.

Run on your SLURM cluster (not Operon) with:
    conda activate argo
    pytest tests/test_lazyslide_api.py -v --tb=short

These tests are organized in dependency order:
    1. Data download & slide opening
    2. Preprocessing (tissue detection, tiling)
    3. Feature extraction (with num_workers/batch_size)
    4. Aggregation (agg_wsi)
    5. Visualization from zarr (no re-extraction)
    6. Scanpy integration on AnnData output

GPU tests are marked with @pytest.mark.gpu and skipped if no CUDA available.
CPU-only tests (resnet50) are used by default for fast validation.

Public data used:
    - GTEX-1117F-0526.svs (small artery slide, ~20K×20K, 253 tiles)
    - GTEx_artery_dataset.csv.gz (multi-slide metadata table)
    - Both from: huggingface.co/datasets/RendeiroLab/LazySlide-data
"""

import pytest
from pathlib import Path

import pandas as pd

pytestmark = pytest.mark.network

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def large_slide(data_dir):
    """Download the larger GTEx small intestine slide (GTEX-11DXX-1626.svs).

    This is a ~39K×52K slide that produces ~19K tiles at 128px.
    Used in LazySlide's feature extraction tutorial.
    Only download if needed (larger file).
    """
    from huggingface_hub import hf_hub_download

    slide_path = hf_hub_download(
        "rendeirolab/lazyslide-data",
        "GTEX-11DXX-1626.svs",
        repo_type="dataset",
        cache_dir=str(data_dir),
        token=False,
    )
    return Path(slide_path)


@pytest.fixture(scope="session")
def gtex_dataset():
    """Download the GTEx artery dataset metadata table.

    Used in LazySlide's multi-slide tutorial for agg_wsi testing.
    """
    from huggingface_hub import hf_hub_download

    table_path = hf_hub_download(
        "rendeirolab/lazyslide-data",
        "GTEx_artery_dataset.csv.gz",
        repo_type="dataset",
        token=False,
    )
    return pd.read_csv(table_path)


# ============================================================================
# 1. Core: Slide Opening
# ============================================================================


class TestSlideOpening:
    """Test wsidata.open_wsi() patterns from reference guide Section 2."""

    def test_open_wsi_basic(self, sample_slide):
        """open_wsi creates a WSIData object with expected properties."""
        from wsidata import open_wsi

        wsi = open_wsi(str(sample_slide))
        assert wsi is not None
        # Should have properties accessible
        assert wsi.properties.mpp is not None
        # wsidata 0.8 exposes shape (W, H) on properties rather than width/height
        w, h = wsi.properties.shape
        assert w > 0 and h > 0
        print(f"  Slide: {w}x{h}, {wsi.properties.mpp:.4f} mpp")

    def test_open_wsi_no_thumbnail(self, sample_slide):
        """open_wsi with attach_thumbnail=False skips thumbnail generation."""
        from wsidata import open_wsi

        wsi = open_wsi(str(sample_slide), attach_thumbnail=False)
        assert wsi is not None
        # Should still work for all downstream operations

    def test_open_wsi_custom_store(self, sample_slide, data_dir):
        """open_wsi with store= saves zarr to custom directory."""
        from wsidata import open_wsi

        store_dir = data_dir / "custom_store"
        store_dir.mkdir(exist_ok=True)
        wsi = open_wsi(str(sample_slide), store=str(store_dir))
        assert wsi is not None


# ============================================================================
# 2. Preprocessing
# ============================================================================


class TestPreprocessing:
    """Test zs.pp.* patterns from reference guide Section 3."""

    def test_find_tissues(self, sample_slide):
        """find_tissues detects tissue regions and stores in shapes."""
        from wsidata import open_wsi
        import lazyslide as zs

        wsi = open_wsi(str(sample_slide), attach_thumbnail=False)
        zs.pp.find_tissues(wsi)

        # Should have tissues in shapes
        assert "tissues" in wsi.shapes
        n_tissues = len(wsi.shapes["tissues"])
        assert n_tissues > 0
        print(f"  Found {n_tissues} tissue regions")

    def test_tile_tissues(self, sample_slide):
        """tile_tissues creates tile grid and stores in shapes."""
        from wsidata import open_wsi
        import lazyslide as zs

        wsi = open_wsi(str(sample_slide), attach_thumbnail=False)
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, tile_px=256, mpp=0.5)

        assert "tiles" in wsi.shapes
        n_tiles = len(wsi.shapes["tiles"])
        assert n_tiles > 0
        print(f"  Created {n_tiles} tiles at 256px / 0.5mpp")

    def test_tile_tissues_with_background_fraction(self, sample_slide):
        """background_fraction parameter filters low-tissue tiles."""
        from wsidata import open_wsi
        import lazyslide as zs

        wsi = open_wsi(str(sample_slide), attach_thumbnail=False)
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, tile_px=256, mpp=0.5, background_fraction=0.5)

        n_tiles = len(wsi.shapes["tiles"])
        assert n_tiles > 0
        print(f"  Created {n_tiles} tiles with background_fraction=0.5")

    def test_write_and_reload(self, sample_slide, data_dir):
        """wsi.write() persists, and re-opening auto-loads from zarr."""
        from wsidata import open_wsi
        import lazyslide as zs

        store_dir = data_dir / "write_test"
        store_dir.mkdir(exist_ok=True)

        # Process and write
        wsi = open_wsi(str(sample_slide), attach_thumbnail=False, store=str(store_dir))
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, tile_px=256, mpp=0.5)
        n_tiles_original = len(wsi.shapes["tiles"])
        wsi.write()

        # Re-open — should auto-load zarr with tissues and tiles
        wsi2 = open_wsi(str(sample_slide), store=str(store_dir))
        assert "tissues" in wsi2.shapes
        assert "tiles" in wsi2.shapes
        assert len(wsi2.shapes["tiles"]) == n_tiles_original
        print(f"  Write/reload preserved {n_tiles_original} tiles")


# ============================================================================
# 3. Feature Extraction
# ============================================================================


class TestFeatureExtraction:
    """Test zs.tl.feature_extraction() patterns from reference guide Section 4.

    Uses resnet50 (no auth, fast on CPU) for basic tests.
    GPU tests use uni2/virchow2 and are marked accordingly.
    """

    def test_feature_extraction_resnet50_cpu(self, sample_slide, data_dir):
        """Feature extraction with resnet50 on CPU (no auth needed)."""
        from wsidata import open_wsi
        import lazyslide as zs

        store_dir = data_dir / "feat_test_resnet"
        store_dir.mkdir(exist_ok=True)

        wsi = open_wsi(str(sample_slide), attach_thumbnail=False, store=str(store_dir))
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, tile_px=256, mpp=0.5)

        zs.tl.feature_extraction(
            wsi,
            model="resnet50",
            device="cpu",
            batch_size=16,
            num_workers=0,  # safe for CPU test
            pbar=False,
        )

        # Verify features stored correctly
        assert "resnet50_tiles" in wsi.tables
        adata = wsi["resnet50_tiles"]
        assert adata.shape[0] > 0  # has tiles
        assert adata.shape[1] > 0  # has features
        print(f"  resnet50: {adata.shape[0]} tiles x {adata.shape[1]} features")

        wsi.write()

    def test_feature_extraction_num_workers(self, sample_slide, data_dir):
        """Verify num_workers parameter is accepted and works."""
        from wsidata import open_wsi
        import lazyslide as zs

        store_dir = data_dir / "feat_test_workers"
        store_dir.mkdir(exist_ok=True)

        wsi = open_wsi(str(sample_slide), attach_thumbnail=False, store=str(store_dir))
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, tile_px=256, mpp=0.5)

        # This is the KEY test — verify num_workers>0 doesn't crash
        zs.tl.feature_extraction(
            wsi,
            model="resnet50",
            device="cpu",
            batch_size=32,
            num_workers=2,  # THE PARAMETER WE NEED TO VERIFY
            pbar=False,
        )

        assert "resnet50_tiles" in wsi.tables
        print(f"  num_workers=2 succeeded: {wsi['resnet50_tiles'].shape}")

    def test_feature_extraction_batch_size(self, sample_slide, data_dir):
        """Verify batch_size parameter affects processing."""
        from wsidata import open_wsi
        import lazyslide as zs

        store_dir = data_dir / "feat_test_batchsize"
        store_dir.mkdir(exist_ok=True)

        wsi = open_wsi(str(sample_slide), attach_thumbnail=False, store=str(store_dir))
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, tile_px=256, mpp=0.5)

        zs.tl.feature_extraction(
            wsi,
            model="resnet50",
            device="cpu",
            batch_size=64,  # larger batch size
            num_workers=0,
            pbar=False,
        )

        assert "resnet50_tiles" in wsi.tables
        print(f"  batch_size=64 succeeded: {wsi['resnet50_tiles'].shape}")

    @pytest.mark.model_download
    def test_multi_model_extraction(self, sample_slide, data_dir):
        """Extract multiple models on same slide, single write."""
        from wsidata import open_wsi
        import lazyslide as zs

        store_dir = data_dir / "feat_test_multi"
        store_dir.mkdir(exist_ok=True)

        wsi = open_wsi(str(sample_slide), attach_thumbnail=False, store=str(store_dir))
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, tile_px=256, mpp=0.5)

        # Extract two models, single preprocess
        for model in ["resnet50", "ctranspath"]:
            zs.tl.feature_extraction(
                wsi,
                model=model,
                device="cpu",
                batch_size=16,
                num_workers=0,
                pbar=False,
            )

        # Both should be in tables
        assert "resnet50_tiles" in wsi.tables
        assert "ctranspath_tiles" in wsi.tables

        wsi.write()
        print(
            f"  Multi-model: resnet50={wsi['resnet50_tiles'].shape}, "
            f"ctranspath={wsi['ctranspath_tiles'].shape}"
        )

    @pytest.mark.gpu
    def test_feature_extraction_uni2_gpu(self, sample_slide, data_dir, has_cuda):
        """Feature extraction with uni2 on GPU (gated, needs HF_TOKEN)."""
        if not has_cuda:
            pytest.skip("No CUDA available")

        from wsidata import open_wsi
        import lazyslide as zs

        store_dir = data_dir / "feat_test_uni2"
        store_dir.mkdir(exist_ok=True)

        wsi = open_wsi(str(sample_slide), attach_thumbnail=False, store=str(store_dir))
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, tile_px=256, mpp=0.5)

        zs.tl.feature_extraction(
            wsi,
            model="uni2",
            amp=True,
            device="cuda",
            num_workers=4,
            batch_size=64,
            pbar=False,
        )

        assert "uni2_tiles" in wsi.tables
        adata = wsi["uni2_tiles"]
        assert adata.shape[1] == 1024  # UNI2 feature dim
        print(f"  uni2 GPU: {adata.shape}")
        wsi.write()


# ============================================================================
# 4. Feature Aggregation
# ============================================================================


class TestFeatureAggregation:
    """Test aggregation patterns from reference guide Sections 5-6."""

    def test_feature_aggregation_default(self, sample_slide, data_dir):
        """zs.tl.feature_aggregation() with default mean pooling."""
        from wsidata import open_wsi
        import lazyslide as zs

        store_dir = data_dir / "agg_test"
        store_dir.mkdir(exist_ok=True)

        wsi = open_wsi(str(sample_slide), attach_thumbnail=False, store=str(store_dir))
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, tile_px=256, mpp=0.5)
        zs.tl.feature_extraction(
            wsi,
            model="resnet50",
            device="cpu",
            batch_size=16,
            num_workers=0,
            pbar=False,
        )

        # Default aggregation
        zs.tl.feature_aggregation(wsi, feature_key="resnet50")

        # Check result is stored (in varm or uns)
        adata = wsi["resnet50_tiles"]
        has_agg = ("agg_slide" in adata.varm) or ("agg_slide" in adata.uns)
        assert has_agg, "Aggregation result not found in varm or uns"
        print(f"  Aggregation stored in: {'varm' if 'agg_slide' in adata.varm else 'uns'}")

        wsi.write()

    def test_agg_wsi_batch(self, sample_slide, data_dir):
        """wsidata.agg_wsi() batch aggregation across slides.

        This is THE critical test — validates the one-liner that replaces
        the entire manual aggregation loop.

        Uses the same slide twice (as if it were two different slides)
        to test the batch aggregation interface.
        """
        from wsidata import open_wsi, agg_wsi
        import lazyslide as zs

        store_dir = data_dir / "agg_wsi_test"
        store_dir.mkdir(exist_ok=True)

        # Process slide and save
        wsi = open_wsi(str(sample_slide), attach_thumbnail=False, store=str(store_dir))
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, tile_px=256, mpp=0.5)
        zs.tl.feature_extraction(
            wsi,
            model="resnet50",
            device="cpu",
            batch_size=16,
            num_workers=0,
            pbar=False,
        )
        zs.tl.feature_aggregation(wsi, feature_key="resnet50")
        wsi.write()

        # Find the zarr path
        zarr_path = None
        for p in store_dir.glob("*.zarr"):
            zarr_path = str(p)
            break

        if zarr_path is None:
            # Try alongside the SVS
            zarr_path = str(sample_slide.with_suffix(".zarr"))

        assert (
            zarr_path is not None and Path(zarr_path).exists()
        ), f"Zarr not found in {store_dir} or alongside SVS"

        # Create a fake multi-slide dataset (same slide twice)
        dataset = pd.DataFrame(
            {
                "slide_id": ["slide_1", "slide_2"],
                "store": [zarr_path, zarr_path],
                "label": ["healthy", "calcified"],
            }
        )

        # THE KEY CALL — agg_wsi
        agg_data = agg_wsi(
            dataset,
            feature_key="resnet50",
            store_col="store",
            agg_key="agg_slide",
        )

        # Validate output
        assert agg_data is not None, "agg_wsi returned None"
        assert hasattr(agg_data, "X"), "agg_wsi result is not AnnData"
        assert agg_data.shape[0] == 2, f"Expected 2 slides, got {agg_data.shape[0]}"
        assert agg_data.shape[1] > 0, "No features in aggregated data"
        print(f"  agg_wsi result: {agg_data.shape} (2 slides x {agg_data.shape[1]} features)")

    def test_agg_wsi_scanpy_integration(self, sample_slide, data_dir):
        """agg_wsi output works directly with scanpy."""
        from wsidata import open_wsi, agg_wsi
        import lazyslide as zs
        import scanpy as sc

        store_dir = data_dir / "agg_scanpy_test"
        store_dir.mkdir(exist_ok=True)

        # Process and write
        wsi = open_wsi(str(sample_slide), attach_thumbnail=False, store=str(store_dir))
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, tile_px=256, mpp=0.5)
        zs.tl.feature_extraction(
            wsi,
            model="resnet50",
            device="cpu",
            batch_size=16,
            num_workers=0,
            pbar=False,
        )
        zs.tl.feature_aggregation(wsi, feature_key="resnet50")
        wsi.write()

        # Find zarr
        zarr_path = None
        for p in store_dir.glob("*.zarr"):
            zarr_path = str(p)
            break
        if zarr_path is None:
            zarr_path = str(sample_slide.with_suffix(".zarr"))

        # Create dataset with 3+ slides (need >=3 for neighbors)
        dataset = pd.DataFrame(
            {
                "slide_id": [f"slide_{i}" for i in range(5)],
                "store": [zarr_path] * 5,
                "label": ["A", "B", "A", "B", "A"],
            }
        )

        agg_data = agg_wsi(dataset, "resnet50", store_col="store", agg_key="agg_slide")

        # Add labels to obs
        agg_data.obs["label"] = dataset["label"].values

        # Scanpy integration — this should work on AnnData. With only 5 "slides",
        # fall back to use_rep='X' so scanpy doesn't try 50-component PCA.
        sc.pp.neighbors(agg_data, n_neighbors=3, use_rep="X")
        sc.tl.umap(agg_data)

        assert "X_umap" in agg_data.obsm
        print(f"  Scanpy UMAP on agg_wsi output: {agg_data.obsm['X_umap'].shape}")


# ============================================================================
# 5. Zarr-Based Visualization (no re-extraction)
# ============================================================================


class TestVisualizationFromZarr:
    """Test that visualization works from pre-computed zarr data."""

    def test_load_precomputed_features(self, sample_slide, data_dir):
        """Opening a slide with existing zarr loads features without GPU."""
        from wsidata import open_wsi
        import lazyslide as zs

        store_dir = data_dir / "viz_test"
        store_dir.mkdir(exist_ok=True)

        # First: process and save
        wsi = open_wsi(str(sample_slide), attach_thumbnail=False, store=str(store_dir))
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, tile_px=256, mpp=0.5)
        zs.tl.feature_extraction(
            wsi,
            model="resnet50",
            device="cpu",
            batch_size=16,
            num_workers=0,
            pbar=False,
        )
        wsi.write()

        # Second: reopen — features should be there without extraction
        wsi2 = open_wsi(str(sample_slide), store=str(store_dir))
        assert "resnet50_tiles" in wsi2.tables
        adata = wsi2["resnet50_tiles"]
        assert adata.shape[0] > 0
        print(f"  Loaded pre-computed features: {adata.shape}")

    def test_plot_from_zarr(self, sample_slide, data_dir):
        """zs.pl.* functions work on pre-computed data."""
        from wsidata import open_wsi
        import lazyslide as zs
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt

        store_dir = data_dir / "viz_plot_test"
        store_dir.mkdir(exist_ok=True)

        # Process and save
        wsi = open_wsi(str(sample_slide), store=str(store_dir))
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, tile_px=256, mpp=0.5)
        zs.tl.feature_extraction(
            wsi,
            model="resnet50",
            device="cpu",
            batch_size=16,
            num_workers=0,
            pbar=False,
        )
        wsi.write()

        # Reopen and plot (NO feature extraction here)
        wsi2 = open_wsi(str(sample_slide), store=str(store_dir))

        # lazyslide 0.10 exposes zs.pl.tissue (singular). Plain-slide view =
        # tissue plot with contours/ids off; tissue overlay = defaults.
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        zs.pl.tissue(wsi2, ax=axes[0], show_contours=False, show_id=False)
        zs.pl.tissue(wsi2, ax=axes[1])
        plt.close(fig)
        print("  Plotting from zarr succeeded (no GPU needed)")


# ============================================================================
# 6. Direct Zarr Read (fallback for custom pooling)
# ============================================================================


class TestDirectZarrRead:
    """Test the fallback pattern of reading X directly from zarr."""

    def test_zarr_x_read(self, sample_slide, data_dir):
        """Read feature matrix directly from zarr without AnnData overhead."""
        from wsidata import open_wsi
        import lazyslide as zs
        import zarr

        store_dir = data_dir / "zarr_read_test"
        store_dir.mkdir(exist_ok=True)

        # Process and save
        wsi = open_wsi(str(sample_slide), attach_thumbnail=False, store=str(store_dir))
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, tile_px=256, mpp=0.5)
        zs.tl.feature_extraction(
            wsi,
            model="resnet50",
            device="cpu",
            batch_size=16,
            num_workers=0,
            pbar=False,
        )
        wsi.write()

        # Find zarr
        zarr_path = None
        for p in store_dir.glob("*.zarr"):
            zarr_path = p
            break
        if zarr_path is None:
            zarr_path = sample_slide.with_suffix(".zarr")

        # Direct zarr read
        adata_zarr_path = zarr_path / "tables" / "resnet50_tiles"
        assert adata_zarr_path.exists(), f"Tables path not found: {adata_zarr_path}"

        z = zarr.open(str(adata_zarr_path), mode="r")
        X = z["X"][:]
        assert X.ndim == 2
        assert X.shape[0] > 0
        assert X.shape[1] > 0

        # Compute mean embedding (what aggregation does)
        mean_embedding = X.mean(axis=0)
        assert mean_embedding.shape == (X.shape[1],)
        print(f"  Direct zarr read: {X.shape}, mean embedding: {mean_embedding.shape}")


# ============================================================================
# 7. Incremental Extraction
# ============================================================================


class TestIncrementalExtraction:
    """Test that adding models to existing zarr works correctly."""

    @pytest.mark.model_download
    def test_add_model_to_existing_zarr(self, sample_slide, data_dir):
        """Extract model A, write, then extract model B on same zarr."""
        from wsidata import open_wsi
        import lazyslide as zs

        store_dir = data_dir / "incremental_test"
        store_dir.mkdir(exist_ok=True)

        # First extraction: resnet50
        wsi = open_wsi(str(sample_slide), attach_thumbnail=False, store=str(store_dir))
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, tile_px=256, mpp=0.5)
        zs.tl.feature_extraction(
            wsi,
            model="resnet50",
            device="cpu",
            batch_size=16,
            num_workers=0,
            pbar=False,
        )
        wsi.write()

        # Second extraction: reopen, add ctranspath
        wsi2 = open_wsi(str(sample_slide), store=str(store_dir))
        assert "resnet50_tiles" in wsi2.tables  # first model persisted

        zs.tl.feature_extraction(
            wsi2,
            model="ctranspath",
            device="cpu",
            batch_size=16,
            num_workers=0,
            pbar=False,
        )
        wsi2.write()

        # Third open: both models should be present
        wsi3 = open_wsi(str(sample_slide), store=str(store_dir))
        assert "resnet50_tiles" in wsi3.tables
        assert "ctranspath_tiles" in wsi3.tables
        print(
            f"  Incremental: resnet50={wsi3['resnet50_tiles'].shape}, "
            f"ctranspath={wsi3['ctranspath_tiles'].shape}"
        )


# ============================================================================
# Pytest configuration
# ============================================================================


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: marks tests requiring CUDA GPU")
