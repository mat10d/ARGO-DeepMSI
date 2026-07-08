"""A2 harvest — concat stain-norm shards into a canonical OAUTHC embedding matrix.

Reads results/embeddings/conch_v1.5_titan_stainnorm/shard_*_of_*.{npy,csv} written by
the SLURM array and writes the canonical embeddings.npy + metadata.csv (OAUTHC-prospective
slides only). The stainnorm_probe scorer merges these with the original conch_v1.5_titan
vectors for every non-OAUTHC site.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("results/embeddings/conch_v1.5_titan_stainnorm")


def main() -> None:
    npys = sorted(glob.glob(str(OUT_DIR / "shard_*_of_*.npy")))
    if not npys:
        raise SystemExit(f"no shard npy files under {OUT_DIR}")
    Xs, metas = [], []
    for p in npys:
        c = p[:-4] + ".csv"
        Xs.append(np.load(p))
        metas.append(pd.read_csv(c))
    X = np.vstack(Xs).astype(np.float32)
    meta = pd.concat(metas, ignore_index=True)
    # De-dup any slide processed twice (shouldn't happen, but be safe: keep first).
    keep = ~meta["slide_id"].duplicated()
    X, meta = X[keep.to_numpy()], meta[keep].reset_index(drop=True)
    np.save(OUT_DIR / "embeddings.npy", X)
    meta.to_csv(OUT_DIR / "metadata.csv", index=False)
    print(f"merged {len(npys)} shards -> {len(meta)} OAUTHC slides, dim {X.shape[1]}")
    print(meta["site"].value_counts().to_dict())


if __name__ == "__main__":
    main()
