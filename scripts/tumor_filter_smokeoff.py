"""Q3-tumor-filter SMOKE-OFF (fixed 20-slide probe set).

Compares two candidate tumor-area filters on the same probe slides:

  (a) GrandQC tissue segmentation — ``zs.seg.tissue(model='grandqc')``. This is
      the tissue-vs-background head; it has NO tumor class, so its "tumor" tile
      set is undefined. We record its tissue coverage of the existing tile grid
      to show it merely re-detects tissue (it cannot isolate tumor).
  (b) CTransPath NCT-CRC-HE 9-class tissue classifier (already trained,
      results/analysis/c5_phase1c/tissue_head_ctranspath_nonorm.joblib; holdout
      TUM recall 0.91). Per-tile argmax over the cached ``ctranspath_tiles``
      features gives a genuine TUM subset.

Decision metric (every probe slide is a known colorectal-cancer resection, i.e.
a known-tumor slide):
  - tumor_localization_recall : fraction of probe slides on which the method
    yields a PROPER tumor-tile subset (0 < tumor_fraction < 1). A method that
    keeps every tile (GrandQC tissue) or finds no tumor scores 0.
  - false_drop_rate@floor     : fraction of known-tumor slides that would be
    dropped at a candidate tumor-fraction floor.

Writes results/data/tumor_smokeoff/{smokeoff_metrics.json, probe_slides.csv}.
The winner (higher tumor_localization_recall) is written to ``winner`` for the
harvest step to read before applying to the full cohort.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("q3_smokeoff")

OUTDIR = Path("results/data/tumor_smokeoff")
HEAD_PATH = Path("results/analysis/c5_phase1c/tissue_head_ctranspath_nonorm.joblib")


def select_probe(cohort_csv: Path, slide_table_csv: Path, n: int, seed: int) -> pd.DataFrame:
    """Deterministic per-site-stratified probe of ``n`` in-clean-set slides."""
    coh = pd.read_csv(cohort_csv)
    coh = coh[coh["in_clean_set"] == 1].copy()
    st = pd.read_csv(slide_table_csv)
    st["slide_id"] = st["FILENAME"].apply(lambda p: Path(p).stem)
    src = st[["slide_id", "FILENAME"]].drop_duplicates("slide_id")
    coh = coh.merge(src, on="slide_id", how="left").rename(columns={"FILENAME": "source"})

    # Sites ordered largest-first so the remainder lands on the best-populated
    # sites; deterministic given a fixed cohort.
    sites = list(coh["site"].value_counts().index)
    per_site = max(1, n // len(sites))
    picks, taken = [], {}
    for site in sites:
        sub = coh[coh["site"] == site].sort_values("slide_id")
        take = min(per_site, len(sub))
        picks.append(sub.sample(n=take, random_state=seed))
        taken[site] = take
    probe = pd.concat(picks, ignore_index=True)
    # Fill the remainder (n - sum) from the largest sites that still have slides.
    deficit = n - len(probe)
    for site in sites:
        if deficit <= 0:
            break
        sub = coh[(coh["site"] == site)
                  & (~coh["slide_id"].isin(probe["slide_id"]))].sort_values("slide_id")
        add = min(deficit, len(sub))
        if add:
            probe = pd.concat([probe, sub.sample(n=add, random_state=seed + 1)],
                              ignore_index=True)
            deficit -= add
    if len(probe) > n:
        probe = probe.sample(n=n, random_state=seed).reset_index(drop=True)
    return probe.sort_values(["site", "slide_id"]).reset_index(drop=True)


def _load_ctp_X(zarr_path: Path) -> np.ndarray | None:
    import anndata as ad
    import zarr as _zarr

    try:
        adata = ad.read_zarr(str(zarr_path / "tables" / "ctranspath_tiles"))
        return np.asarray(adata.X)
    except Exception:
        try:
            z = _zarr.open_group(str(zarr_path / "tables" / "ctranspath_tiles"), mode="r")
            return np.asarray(z["X"])
        except Exception as e:
            log.warning(f"ctranspath read failed for {zarr_path.name}: {e}")
            return None


def ctp_tumor(zarr_path: Path, clf, labels: list[str]) -> dict | None:
    """CTransPath per-tile TUM fraction for one slide."""
    X = _load_ctp_X(zarr_path)
    if X is None or X.size == 0:
        return None
    tum_id = labels.index("TUM")
    yhat = clf.predict_proba(X).argmax(axis=1)
    n = int(len(yhat))
    n_tum = int((yhat == tum_id).sum())
    return {"n_tiles": n, "n_tum": n_tum, "tumor_fraction": n_tum / n if n else 0.0}


def grandqc_tissue(source: Path, device: str) -> dict:
    """GrandQC tissue coverage of the existing tile grid for one slide.

    Runs ``zs.seg.tissue(model='grandqc')`` and measures the fraction of the
    cached tile centroids that fall inside the GrandQC tissue polygons. GrandQC
    has no tumor class, so ``tumor_fraction`` is undefined (None).
    """
    import lazyslide as zs
    from shapely.geometry import Point
    from wsidata import open_wsi

    source = Path(source)
    wsi = open_wsi(str(source), store=str(source.parent), attach_thumbnail=False)
    if "tiles" not in wsi.shapes:
        return {"status": "no_tiles", "tumor_fraction": None}
    tiles = wsi.shapes["tiles"]
    n_tiles = int(len(tiles))

    zs.seg.tissue(wsi, model="grandqc", device=device, key_added="gqc_tissue")
    if "gqc_tissue" not in wsi.shapes or len(wsi.shapes["gqc_tissue"]) == 0:
        return {"status": "ok", "n_tiles": n_tiles, "n_tissue_tiles": 0,
                "tissue_tile_fraction": 0.0, "n_gqc_polys": 0,
                "tumor_fraction": None, "has_tumor_class": False}

    gqc_union = wsi.shapes["gqc_tissue"].geometry.unary_union
    cents = tiles.geometry.centroid
    inside = int(sum(gqc_union.contains(Point(c.x, c.y)) for c in cents))
    return {
        "status": "ok",
        "n_tiles": n_tiles,
        "n_tissue_tiles": inside,
        "tissue_tile_fraction": inside / n_tiles if n_tiles else 0.0,
        "n_gqc_polys": int(len(wsi.shapes["gqc_tissue"])),
        "tumor_fraction": None,        # GrandQC tissue head has no tumor class
        "has_tumor_class": False,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", default="results/data/cohort_clean.csv", type=Path)
    p.add_argument("--slide-table", default="results/data/slide_table_pyramidal.csv", type=Path)
    p.add_argument("--n-probe", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--floor", type=float, default=0.01, help="candidate tumor-fraction floor")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    probe = select_probe(a.cohort, a.slide_table, a.n_probe, a.seed)
    probe.to_csv(OUTDIR / "probe_slides.csv", index=False)
    log.info(f"probe: {len(probe)} slides across {probe['site'].nunique()} sites")

    head = joblib.load(HEAD_PATH)
    clf, labels = head["clf"], head["labels"]

    rows = []
    for _, r in probe.iterrows():
        sid = r["slide_id"]
        zarr_path = Path(r["source"]).with_suffix(".zarr")
        rec = {"slide_id": sid, "site": r["site"]}
        # candidate (b) CTransPath
        ctp = ctp_tumor(zarr_path, clf, labels)
        if ctp is None:
            rec["ctp_status"] = "no_features"
            rec["ctp_tumor_fraction"] = np.nan
        else:
            rec["ctp_status"] = "ok"
            rec["ctp_n_tiles"] = ctp["n_tiles"]
            rec["ctp_n_tum"] = ctp["n_tum"]
            rec["ctp_tumor_fraction"] = ctp["tumor_fraction"]
        # candidate (a) GrandQC tissue
        try:
            g = grandqc_tissue(Path(r["source"]), a.device)
            rec["gqc_status"] = g.get("status", "ok")
            rec["gqc_tissue_tile_fraction"] = g.get("tissue_tile_fraction", np.nan)
            rec["gqc_n_polys"] = g.get("n_gqc_polys", 0)
            rec["gqc_tumor_fraction"] = np.nan  # no tumor class
        except Exception as e:
            log.warning(f"grandqc failed on {sid}: {e}")
            rec["gqc_status"] = f"fail: {type(e).__name__}"
            rec["gqc_tissue_tile_fraction"] = np.nan
            rec["gqc_tumor_fraction"] = np.nan
        rows.append(rec)
        log.info(f"{sid} [{r['site']}] ctp_tf="
                 f"{rec.get('ctp_tumor_fraction')} gqc_tissue_tf="
                 f"{rec.get('gqc_tissue_tile_fraction')}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "probe_scores.csv", index=False)

    floor = a.floor
    ctp_ok = df["ctp_tumor_fraction"].notna()
    n_known = int(ctp_ok.sum())

    def _proper_subset_recall(frac_col: str) -> float:
        f = df[frac_col].dropna()
        if len(f) == 0:
            return 0.0
        return float(((f > 0) & (f < 1)).mean())

    ctp_recall = _proper_subset_recall("ctp_tumor_fraction")
    # GrandQC tissue head: no tumor class -> tumor-tile localization impossible.
    gqc_recall = 0.0

    ctp_false_drop = float((df["ctp_tumor_fraction"] < floor).mean()) if n_known else float("nan")

    metrics = {
        "task": "Q3-tumor-filter smoke-off",
        "n_probe": int(len(df)),
        "n_sites": int(df["site"].nunique()),
        "candidate_floor": floor,
        "candidates": {
            "grandqc_tissue": {
                "model": "zs.seg.tissue(model='grandqc')",
                "has_tumor_class": False,
                "tumor_localization_recall": gqc_recall,
                "median_tissue_tile_fraction": float(
                    df["gqc_tissue_tile_fraction"].median()),
                "note": "tissue-vs-background only; cannot isolate tumor tiles",
            },
            "ctranspath": {
                "model": "NCT-CRC-HE 9-class LR head on ctranspath features",
                "has_tumor_class": True,
                "holdout_tum_recall": 0.910,
                "tumor_localization_recall": ctp_recall,
                "median_tumor_fraction": float(df["ctp_tumor_fraction"].median()),
                "false_drop_rate_at_floor": ctp_false_drop,
            },
        },
        "winner": "ctranspath" if ctp_recall > gqc_recall else "grandqc_tissue",
        "winner_rationale": (
            "Only CTransPath exposes a tumor (TUM) class; GrandQC tissue "
            "segmentation is background removal and cannot localize tumor "
            "tiles, so it scores 0 tumor-localization recall."
        ),
    }
    (OUTDIR / "smokeoff_metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info(json.dumps(metrics, indent=2))
    log.info(f"WINNER: {metrics['winner']}")


if __name__ == "__main__":
    main()
