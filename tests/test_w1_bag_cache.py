from __future__ import annotations

import numpy as np
import pandas as pd

from argo_deepmsi.w1 import CachedCTransPathBagStore


def test_cached_bag_store_is_reproducible_and_epoch_variable(tmp_path):
    data = np.arange(40 * 3, dtype=np.float32).reshape(40, 3)
    data_path = tmp_path / "bags.npy"
    np.save(data_path, data)
    cache_index = pd.DataFrame(
        {
            "slide_id": ["a", "b"],
            "start": [0, 20],
            "length": [20, 20],
        }
    )
    cache_index_path = tmp_path / "bags_index.csv"
    cache_index.to_csv(cache_index_path, index=False)
    slide_index = pd.DataFrame({"slide_id": ["a", "b"]})
    store = CachedCTransPathBagStore(
        data_path, cache_index_path, slide_index, max_tiles=7, seed=42
    )
    first = store.read(0, epoch=1)
    repeat = store.read(0, epoch=1)
    other = store.read(0, epoch=2)
    np.testing.assert_array_equal(first, repeat)
    assert first.shape == (7, 3)
    assert not np.array_equal(first, other)
