"""Retrain slide_attention_mil on the QC-clean cohort.

Reuses the model and training loop from ``scripts/slide_attention_mil.py``
but runs only the previously-best single config
(``embedding=none``, ``feature_set=wagner+meta``) on a Wagner slide score
table filtered through ``results/data/problem_slides.csv``.

Output:
    results/scorers/slide_attention_mil/patient_scores_clean.csv

Usage:
    sbatch scripts/slide_attention_mil_qc_retrain.sh
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

# Make the sibling script importable as a module.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import slide_attention_mil as sam  # noqa: E402

from argo_deepmsi.eval.cohort import load_qc_exclusion  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = Path("results/scorers/slide_attention_mil")
OUT_CACHE = OUT_DIR / "patient_scores_clean.csv"
QC_CSV = Path("results/data/problem_slides.csv")
FILTERED_WAGNER = Path("results/scorers/slide_attention_mil/_wagner_clean.csv")

# Best config from results/analysis/c5_phase2/ablation_results.csv
BEST_EMBEDDING = "conch_v1.5_mean"   # placeholder; not used when "embedding": False
BEST_FEATURE_SET = {"wagner": True, "embedding": False, "metadata": True}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    excluded = load_qc_exclusion(QC_CSV)
    log.info(f"QC exclusion: {len(excluded)} slides flagged")

    wagner = pd.read_csv(sam.WAGNER_SLIDES)
    before = len(wagner)
    wagner_clean = wagner[~wagner["slide_id"].isin(excluded)].reset_index(drop=True)
    log.info(f"Wagner slides: {before} → {len(wagner_clean)}")
    wagner_clean.to_csv(FILTERED_WAGNER, index=False)

    # Monkey-patch the WAGNER_SLIDES constant so load_patient_bags uses our filtered file.
    sam.WAGNER_SLIDES = FILTERED_WAGNER

    bags, labels, patients, slide_info = sam.load_patient_bags(
        emb_model=BEST_EMBEDDING, feature_config=BEST_FEATURE_SET
    )
    cv = sam.cross_validate(bags, labels, patients, slide_info)

    oof = pd.DataFrame({
        "patient_id": cv["patients"],
        "y": cv["labels"],
        "p_msih": cv["oof_preds"],
    })
    oof.to_csv(OUT_CACHE, index=False)
    log.info(f"Wrote {OUT_CACHE}  ({len(oof)} patients)")
    log.info(f"Retrained patient AUROC (clean cohort): {cv['auroc']:.3f}")

    # Per-site
    for site, metrics in cv["per_site"].items():
        log.info(f"  {site:<22s}  n={metrics['n']:<4d}  AUROC={metrics['auroc']:.3f}")


if __name__ == "__main__":
    main()
