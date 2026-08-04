from __future__ import annotations

import json

import numpy as np
import pandas as pd

from argo_deepmsi.w1_peft import (
    FixedRawTileStore,
    MixedBagStore,
    _tile_rows_in_feature_order,
    fixed_raw_tile_cache_ready,
)


def test_tile_geometry_uses_feature_row_order_without_requiring_tile_id():
    class FakeWSI:
        shapes = {
            "tiles": pd.DataFrame(
                {"geometry": ["third", "first", "second"]},
                index=[8, 2, 5],
            )
        }

    tiles = _tile_rows_in_feature_order(FakeWSI())
    assert tiles.index.tolist() == [0, 1, 2]
    assert tiles["geometry"].tolist() == ["third", "first", "second"]


def test_fixed_raw_cache_is_ready_only_after_all_artifacts_publish(tmp_path):
    np.save(tmp_path / "images.npy", np.zeros((1,), dtype=np.uint8))
    assert not fixed_raw_tile_cache_ready(tmp_path)
    pd.DataFrame({"slide_id": ["a"]}).to_csv(tmp_path / "index.csv", index=False)
    assert not fixed_raw_tile_cache_ready(tmp_path)
    (tmp_path / "metadata.json").write_text("{}")
    assert fixed_raw_tile_cache_ready(tmp_path)


def test_mixed_bag_store_keeps_features_and_source_indices_aligned(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    features = np.arange(30 * 4, dtype=np.float32).reshape(30, 4)
    source = np.arange(100, 130, dtype=np.int64)
    np.save(cache_dir / "bags.npy", features)
    np.save(cache_dir / "tile_indices.npy", source)
    pd.DataFrame(
        {"slide_id": ["a", "b"], "start": [0, 15], "length": [15, 15]}
    ).to_csv(cache_dir / "bags_index.csv", index=False)
    slide_index = pd.DataFrame({"slide_id": ["a", "b"]})
    store = MixedBagStore(cache_dir, slide_index, seed=42)
    sample = store.read(0, epoch=3, bag_tiles=9, raw_tiles=4)
    repeat = store.read(0, epoch=3, bag_tiles=9, raw_tiles=4)
    assert sample.cached_features.shape == (9, 4)
    assert sample.raw_positions.shape == (4,)
    np.testing.assert_array_equal(sample.cached_features, repeat.cached_features)
    np.testing.assert_array_equal(sample.raw_positions, repeat.raw_positions)
    recovered = sample.cached_features[sample.raw_positions, 0] // 4 + 100
    np.testing.assert_array_equal(recovered, sample.source_tile_indices)


def test_fixed_raw_store_forces_raw_positions_into_sampled_bag(tmp_path):
    feature_dir = tmp_path / "features"
    raw_dir = tmp_path / "raw"
    feature_dir.mkdir()
    raw_dir.mkdir()
    features = np.arange(20 * 4, dtype=np.float32).reshape(20, 4)
    source = np.arange(100, 120, dtype=np.int64)
    np.save(feature_dir / "bags.npy", features)
    np.save(feature_dir / "tile_indices.npy", source)
    pd.DataFrame({"slide_id": ["a"], "start": [0], "length": [20]}).to_csv(
        feature_dir / "bags_index.csv", index=False
    )
    images = np.zeros((2, 256, 256, 3), dtype=np.uint8)
    images[1] = 7
    np.save(raw_dir / "images.npy", images)
    pd.DataFrame(
        {
            "slide_id": ["a"],
            "start": [0],
            "length": [2],
            "reservoir_positions": [json.dumps([3, 17])],
            "source_tile_indices": [json.dumps([103, 117])],
        }
    ).to_csv(raw_dir / "index.csv", index=False)
    slide_index = pd.DataFrame({"slide_id": ["a"]})
    store = FixedRawTileStore(feature_dir, raw_dir, slide_index, seed=42)
    sample = store.read(0, epoch=5, bag_tiles=7, raw_tiles=2)
    assert sample.cached_features.shape == (7, 4)
    np.testing.assert_array_equal(
        sample.cached_features[sample.raw_positions, 0] // 4 + 100,
        sample.source_tile_indices,
    )
    assert sample.raw_images.shape == (2, 256, 256, 3)
    assert sample.raw_images[1].mean() == 7
