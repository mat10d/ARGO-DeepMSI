# Wagner Error Anatomy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Anatomize the frozen Wagner champion's errors on the 217-patient / 803-slide primary cohort and map each error cause to a concrete circumvention lever.

**Architecture:** A pure-function core (`argo_deepmsi/eval/error_anatomy.py`) builds a patient-resolution error ledger from cached artifacts, applies deterministic attribution rules (Stage A), runs the paired-scan natural experiment (Stage B), and synthesizes the recipe. An additive attention-export hook on the frozen Wagner model (`argo_deepmsi/models/wagner.py`) powers a GPU introspection pass over only the unexplained error subset (Stage C).

**Tech Stack:** Python 3.11, pandas, numpy, scikit-learn, matplotlib, torch (Stage C only), LazySlide/wsidata (Stage C only), pytest, SLURM (nvidia-A6000-20).

## Global Constraints

- **No autonomous commits (user invariant):** where a step says "Commit", instead `git add` the files and run the gate; defer the actual `git commit` to the user's `/commit`. Never run `git commit` unless the user explicitly asks.
- **Reuse existing helpers, do not re-implement:** patient aggregation via `argo_deepmsi.eval.metrics` (`max_sqrtn = v.max()/np.sqrt(len(v))`) or `argo_deepmsi.eval.reliability_weight.patient_max_sqrtn`; operating point via `argo_deepmsi.eval.screening.threshold_at_sensitivity` and `sensitivity_specificity_at_threshold`.
- **Frozen Wagner is immutable:** the attention hook must be additive and off by default; a frozen-equivalence test guards that `p_msih` is numerically unchanged when introspection is off.
- **Primary estimand:** 217 patients / 803 slides / 47 MSI-H. Operating point: sensitivity 0.95.
- **Thresholds are module constants** at the top of `error_anatomy.py`, recorded in the output manifest: `TUMOR_FLOOR = 0.01`, `ARTIFACT_CEILING = 0.5`, `CMO_SCORE_BAND` (fraction of assay cutoff, default ±0.10 of the MSI cutoff), `CONFIDENT_MARGIN = 0.15` (score distance past the threshold to count as confident-wrong).
- **Determinism:** Stage A + B use no RNG; ledgers must reproduce byte-stable from cached inputs.
- **Outputs root:** `results/analysis/error_anatomy/`. Docs: `docs/experiments/E1..E4-*.md`.
- **Env:** `eval "$(conda shell.bash hook)" && conda activate argo`; call tools via `python -m` (not bare pytest).
- **Style:** ruff line length 100, Google docstrings, match surrounding file conventions.

---

### Task 1: Error-ledger builder (Stage A data join + error classes)

**Files:**
- Create: `argo_deepmsi/eval/error_anatomy.py`
- Test: `tests/test_error_anatomy.py`

**Interfaces:**
- Consumes: `argo_deepmsi.eval.metrics` (agg), `argo_deepmsi.eval.screening.threshold_at_sensitivity`, `sensitivity_specificity_at_threshold`.
- Produces:
  - `build_error_ledger(scores_df, cohort_df, clinical_df, *, target_sensitivity=0.95, threshold=None) -> tuple[pd.DataFrame, pd.DataFrame]`
    returns `(patient_ledger, slide_detail)`. When `threshold` is given it overrides the
    sens-derived one (Stages B/C reuse the same threshold; tests inject it for determinism). `patient_ledger` has columns
    `patient_id, y, wagner_patient_score, n_slides, rank_residual, error_class, confident_wrong`
    plus merged covariates `site, processing_site, stain_location, imaging_site, label_source,
    label_certainty, cmo_msi_score, msi_method, mmr_cmo_concordance, n_tiles, tumor_fraction,
    artifact_fraction`. `error_class ∈ {TP,TN,FP,FN}`. `slide_detail` has
    `slide_id, patient_id, p_msih, y, processing_site, stain_location, n_tiles, tumor_fraction, artifact_fraction`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_error_anatomy.py
import numpy as np
import pandas as pd
from argo_deepmsi.eval.error_anatomy import build_error_ledger


def _tiny_inputs():
    # P1: two MSS slides, low scores -> TN. P2: one MSI-H slide, low score -> FN (confident miss).
    scores = pd.DataFrame({
        "slide_id": ["s1", "s2", "s3"],
        "patient_id": ["P1", "P1", "P2"],
        "site": ["UITH", "UITH", "OAUTHC"],
        "y": [0, 0, 1],
        "p_msih": [0.10, 0.20, 0.05],
        "stain_location": ["UITH", "UITH", "OAUTHC"],
    })
    cohort = pd.DataFrame({
        "patient_id": ["P1", "P2"],
        "processing_site": ["UITH", "OAUTHC"],
        "n_tiles": [100, 80],
        "tumor_fraction": [0.5, 0.4],
        "artifact_fraction": [0.1, 0.2],
        "label_source": ["prospective_cmo", "prospective_cmo"],
        "label_certainty": ["definite", "definite"],
        "cmo_msi_score": [1.0, 30.0],
        "msi_status_mmr": [np.nan, np.nan],
        "cmo_msi_status": ["MSS", "MSI-H"],
    })
    clinical = pd.DataFrame({
        "PATIENT": ["P1", "P2"],
        "msi_method": ["PCR", "PCR"],
        "slide_staining_site": ["UITH", "OAUTHC"],
        "slide_imaging_site": ["UITH", "OAUTHC"],
    })
    return scores, cohort, clinical


def test_build_error_ledger_aggregation_and_classes():
    scores, cohort, clinical = _tiny_inputs()
    # inject a threshold so all four error classes are deterministic
    ledger, detail = build_error_ledger(scores, cohort, clinical, threshold=0.5)
    p1 = ledger.set_index("patient_id").loc["P1"]
    # max/sqrt(n): max(0.10,0.20)/sqrt(2) = 0.1414 < 0.5, y=0 -> TN
    assert abs(p1["wagner_patient_score"] - 0.20 / np.sqrt(2)) < 1e-9
    assert p1["n_slides"] == 2
    assert p1["error_class"] == "TN"
    p2 = ledger.set_index("patient_id").loc["P2"]
    # score 0.05 < 0.5, y=1 -> FN (confident: residual < -CONFIDENT_MARGIN)
    assert p2["error_class"] == "FN"
    assert bool(p2["confident_wrong"]) is True
    assert len(detail) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_error_anatomy.py::test_build_error_ledger_aggregation_and_classes -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError: cannot import name 'build_error_ledger'`.

- [ ] **Step 3: Write minimal implementation**

```python
# argo_deepmsi/eval/error_anatomy.py
"""Wagner error anatomy — ledger, attribution, paired-scan, synthesis.

Diagnoses *which* patients/slides the frozen Wagner champion gets wrong and *why*,
then maps each cause to a circumvention lever. Pure functions over dataframes; I/O
lives in the entrypoint (Task 3). See docs/superpowers/specs/2026-08-04-wagner-error-anatomy-design.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from argo_deepmsi.eval.screening import (
    sensitivity_specificity_at_threshold,
    threshold_at_sensitivity,
)

# --- attribution constants (recorded in the manifest) ---
TUMOR_FLOOR = 0.01
ARTIFACT_CEILING = 0.5
CMO_SCORE_BAND = 0.10
CONFIDENT_MARGIN = 0.15


def build_error_ledger(scores_df, cohort_df, clinical_df, *, target_sensitivity=0.95,
                       threshold=None):
    """Join Wagner slide scores + covariates into a patient ledger and slide detail.

    Returns (patient_ledger, slide_detail). If `threshold` is None it is derived from
    `target_sensitivity`; otherwise the given threshold is used (Stages B/C reuse it).
    """
    s = scores_df.copy()
    # patient score = max/sqrt(n) over the patient's slides
    agg = s.groupby("patient_id").agg(
        y=("y", "max"),
        n_slides=("p_msih", "size"),
        _max=("p_msih", "max"),
        site=("site", "first"),
        stain_location=("stain_location", "first"),
    )
    agg["wagner_patient_score"] = agg["_max"] / np.sqrt(agg["n_slides"])
    agg = agg.drop(columns="_max").reset_index()

    thr = float(threshold) if threshold is not None else threshold_at_sensitivity(
        agg["y"].to_numpy(), agg["wagner_patient_score"].to_numpy(), target_sensitivity)
    pred = (agg["wagner_patient_score"] >= thr).astype(int)
    y = agg["y"].astype(int)
    agg["error_class"] = np.select(
        [(y == 1) & (pred == 1), (y == 0) & (pred == 0),
         (y == 0) & (pred == 1), (y == 1) & (pred == 0)],
        ["TP", "TN", "FP", "FN"],
    )
    agg["rank_residual"] = agg["wagner_patient_score"] - thr
    agg["confident_wrong"] = (
        ((agg["error_class"] == "FP") & (agg["rank_residual"] > CONFIDENT_MARGIN))
        | ((agg["error_class"] == "FN") & (agg["rank_residual"] < -CONFIDENT_MARGIN))
    )

    cov = cohort_df.rename(columns={}).copy()
    clin = clinical_df.rename(columns={
        "PATIENT": "patient_id",
        "slide_staining_site": "imaging_stain_site",
        "slide_imaging_site": "imaging_site",
    })
    clin["mmr_cmo_concordance"] = _mmr_cmo_concordance(cohort_df)
    ledger = (agg.merge(cov, on="patient_id", how="left", suffixes=("", "_cov"))
                 .merge(clin[["patient_id", "msi_method", "imaging_site", "mmr_cmo_concordance"]],
                        on="patient_id", how="left"))
    ledger.attrs["threshold"] = float(thr)

    detail = s.merge(
        cohort_df[["patient_id", "n_tiles", "tumor_fraction", "artifact_fraction",
                   "processing_site"]],
        on="patient_id", how="left")
    return ledger, detail


def _mmr_cmo_concordance(cohort_df):
    """True where CMO status and MMR status disagree on MSI-H (label-suspect signal)."""
    def norm(v):
        v = str(v).upper()
        return "MSIH" if "MSI" in v and "MSS" not in v else ("MSS" if "MSS" in v else None)
    cmo = cohort_df["cmo_msi_status"].map(norm)
    mmr = cohort_df["msi_status_mmr"].map(norm)
    return ~((cmo == mmr) | cmo.isna() | mmr.isna())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_error_anatomy.py::test_build_error_ledger_aggregation_and_classes -v`
Expected: PASS

- [ ] **Step 5: Stage + gate (no commit — project rule)**

```bash
git add argo_deepmsi/eval/error_anatomy.py tests/test_error_anatomy.py
python -m pytest tests/test_error_anatomy.py -q
```

---

### Task 2: Attribution rules (Stage A buckets → circumvention levers)

**Files:**
- Modify: `argo_deepmsi/eval/error_anatomy.py`
- Test: `tests/test_error_anatomy.py`

**Interfaces:**
- Consumes: `build_error_ledger` output (Task 1).
- Produces:
  - `attribute_causes(ledger: pd.DataFrame) -> pd.DataFrame` — adds `attributed_cause`,
    `secondary_cause`, `circumvention_lever`. Cause vocabulary:
    `{"label-suspect","low-tumor-content","high-artifact","borderline-score","site-structured",
    "unexplained","correct"}`. Only rows whose `error_class ∈ {FP,FN}` get a non-`correct` cause.
    Priority order: label-suspect > low-tumor-content > high-artifact > borderline-score >
    site-structured > unexplained.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_error_anatomy.py  (append)
from argo_deepmsi.eval.error_anatomy import attribute_causes


def _err_row(**kw):
    base = dict(error_class="FN", confident_wrong=True, label_certainty="definite",
                mmr_cmo_concordance=False, cmo_msi_score=30.0, tumor_fraction=0.5,
                artifact_fraction=0.1, rank_residual=-0.3, site="OAUTHC")
    base.update(kw)
    return base


def test_attribution_priority_and_levers():
    df = pd.DataFrame([
        _err_row(label_certainty="indeterminate_as_mss"),           # -> label-suspect
        _err_row(tumor_fraction=0.0),                               # -> low-tumor-content
        _err_row(artifact_fraction=0.9),                           # -> high-artifact
        _err_row(confident_wrong=False, rank_residual=-0.02),      # -> borderline-score
        dict(error_class="TN", confident_wrong=False),             # -> correct
    ])
    out = attribute_causes(df)
    assert list(out["attributed_cause"]) == [
        "label-suspect", "low-tumor-content", "high-artifact", "borderline-score", "correct"]
    assert out.loc[0, "circumvention_lever"].startswith("re-adjudicat")
    assert out.loc[4, "circumvention_lever"] == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_error_anatomy.py::test_attribution_priority_and_levers -v`
Expected: FAIL (`cannot import name 'attribute_causes'`).

- [ ] **Step 3: Write minimal implementation**

```python
# argo_deepmsi/eval/error_anatomy.py  (append)

_LEVERS = {
    "label-suspect": "re-adjudicate label (molecular/pathology worklist); label-certainty re-run",
    "low-tumor-content": "re-tile / stronger tumor detection (QC-refix worklist)",
    "high-artifact": "artifact removal / re-scan (QC-refix worklist)",
    "borderline-score": "operating-point / selective abstention choice",
    "site-structured": "Stage B: acquisition vs biology",
    "unexplained": "Stage C: spatial attention introspection",
    "correct": "",
}


def _cause_for_row(r) -> str:
    if r["error_class"] not in ("FP", "FN"):
        return "correct"
    near_cutoff = abs(float(r.get("cmo_msi_score", np.nan)) - _MSI_CUTOFF) <= CMO_SCORE_BAND * _MSI_CUTOFF \
        if pd.notna(r.get("cmo_msi_score", np.nan)) else False
    if r.get("label_certainty") == "indeterminate_as_mss" or bool(r.get("mmr_cmo_concordance")) or near_cutoff:
        return "label-suspect"
    if float(r.get("tumor_fraction", 1.0)) <= TUMOR_FLOOR:
        return "low-tumor-content"
    if float(r.get("artifact_fraction", 0.0)) >= ARTIFACT_CEILING:
        return "high-artifact"
    if not bool(r.get("confident_wrong", False)):
        return "borderline-score"
    return "unexplained"  # site-structured is assigned in aggregate (Task 3), else unexplained


_MSI_CUTOFF = 20.0  # cmo_msi_score assay cutoff; verified against ingestion in Task 3 manifest


def attribute_causes(ledger: pd.DataFrame) -> pd.DataFrame:
    out = ledger.copy()
    out["attributed_cause"] = out.apply(_cause_for_row, axis=1)
    out["secondary_cause"] = ""
    out["circumvention_lever"] = out["attributed_cause"].map(_LEVERS).fillna("")
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_error_anatomy.py::test_attribution_priority_and_levers -v`
Expected: PASS

- [ ] **Step 5: Stage + gate**

```bash
git add argo_deepmsi/eval/error_anatomy.py tests/test_error_anatomy.py
python -m pytest tests/test_error_anatomy.py -q
```

**Note for implementer:** confirm `_MSI_CUTOFF` against `argo_deepmsi/data_ingestion.py` (the cmo_msi_score → MSI-H rule) and record the verified value in the Task-3 manifest. If ingestion uses a different cutoff, update the constant.

---

### Task 3: Stage A outputs — Pareto, per-site, worklists, entrypoint

**Files:**
- Modify: `argo_deepmsi/eval/error_anatomy.py`
- Create: `argo_deepmsi/eval/__main__.py` is NOT used — add `run_stage_a` + a `main()` guarded block in `error_anatomy.py`
- Test: `tests/test_error_anatomy.py`
- Create (doc, by implementer after first real run): `docs/experiments/E1-wagner-error-anatomy-A.md`

**Interfaces:**
- Consumes: `build_error_ledger`, `attribute_causes`.
- Produces:
  - `assign_site_structured(ledger) -> ledger` — reclassifies `unexplained` rows to
    `site-structured` where the row's site has an error rate above the cohort mean by a margin.
  - `pareto(ledger) -> pd.DataFrame` — cause × count, overall and OAUTHC-first.
  - `run_stage_a(outdir="results/analysis/error_anatomy") -> dict` — reads the canonical CSVs,
    builds ledger, attributes, writes `A_error_ledger.csv`, `A_slide_detail.csv`,
    `A_pareto.csv`, `A_pareto.png`, `A_per_site_decomposition.csv`,
    `worklist_label_adjudication.csv`, `worklist_qc_refix.csv`, `A_manifest.json`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_error_anatomy.py  (append)
from argo_deepmsi.eval.error_anatomy import assign_site_structured, pareto


def test_site_structured_and_pareto():
    df = pd.DataFrame({
        "attributed_cause": ["unexplained", "unexplained", "correct", "correct"],
        "error_class": ["FN", "FP", "TN", "TP"],
        "site": ["OAUTHC", "OAUTHC", "UITH", "UITH"],
        "circumvention_lever": ["", "", "", ""],
    })
    out = assign_site_structured(df)
    # OAUTHC has 100% error among its rows -> above cohort mean -> site-structured
    assert set(out.loc[out["site"] == "OAUTHC", "attributed_cause"]) == {"site-structured"}
    p = pareto(out)
    assert p.loc[p["attributed_cause"] == "site-structured", "count"].iloc[0] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_error_anatomy.py::test_site_structured_and_pareto -v`
Expected: FAIL (`cannot import name 'assign_site_structured'`).

- [ ] **Step 3: Write minimal implementation**

```python
# argo_deepmsi/eval/error_anatomy.py  (append)
import json
from pathlib import Path


def assign_site_structured(ledger: pd.DataFrame, *, margin: float = 0.10) -> pd.DataFrame:
    out = ledger.copy()
    is_err = out["error_class"].isin(["FP", "FN"])
    site_err = out.assign(_e=is_err).groupby("site")["_e"].mean()
    cohort_mean = float(is_err.mean())
    hot_sites = site_err[site_err > cohort_mean + margin].index
    mask = (out["attributed_cause"] == "unexplained") & out["site"].isin(hot_sites)
    out.loc[mask, "attributed_cause"] = "site-structured"
    out.loc[mask, "circumvention_lever"] = _LEVERS["site-structured"]
    return out


def pareto(ledger: pd.DataFrame) -> pd.DataFrame:
    err = ledger[ledger["attributed_cause"] != "correct"]
    overall = err["attributed_cause"].value_counts().rename("count").reset_index(names="attributed_cause")
    overall["scope"] = "overall"
    oau = err[err["site"] == "OAUTHC"]["attributed_cause"].value_counts().rename("count").reset_index(names="attributed_cause")
    oau["scope"] = "OAUTHC"
    return pd.concat([overall, oau], ignore_index=True)


def run_stage_a(outdir: str = "results/analysis/error_anatomy") -> dict:
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv("results/scorers/wagner_zeroshot/slide_scores.csv")
    cohort = pd.read_csv("results/data/cohort_clean.csv")
    clinical = pd.read_csv("results/data/clinical_table.csv")
    ledger, detail = build_error_ledger(scores, cohort, clinical)
    ledger = assign_site_structured(attribute_causes(ledger))
    ledger.to_csv(out / "A_error_ledger.csv", index=False)
    detail.to_csv(out / "A_slide_detail.csv", index=False)
    p = pareto(ledger); p.to_csv(out / "A_pareto.csv", index=False)
    _plot_pareto(p, out / "A_pareto.png")
    _per_site(ledger).to_csv(out / "A_per_site_decomposition.csv", index=False)
    _worklist_label(ledger).to_csv(out / "worklist_label_adjudication.csv", index=False)
    _worklist_qc(ledger).to_csv(out / "worklist_qc_refix.csv", index=False)
    manifest = {"threshold": ledger.attrs.get("threshold"), "msi_cutoff": _MSI_CUTOFF,
                "tumor_floor": TUMOR_FLOOR, "artifact_ceiling": ARTIFACT_CEILING,
                "cmo_score_band": CMO_SCORE_BAND, "confident_margin": CONFIDENT_MARGIN,
                "n_patients": int(len(ledger)), "n_errors": int((ledger["attributed_cause"] != "correct").sum())}
    (out / "A_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _per_site(ledger):
    g = ledger.assign(_e=ledger["error_class"].isin(["FP", "FN"]))
    return g.groupby("site").agg(n=("patient_id", "size"), n_err=("_e", "sum")).reset_index()


def _worklist_label(ledger):
    cols = ["patient_id", "site", "y", "wagner_patient_score", "cmo_msi_score", "msi_method",
            "label_source", "label_certainty", "mmr_cmo_concordance", "circumvention_lever"]
    w = ledger[ledger["attributed_cause"] == "label-suspect"]
    return w[[c for c in cols if c in w.columns]]


def _worklist_qc(ledger):
    cols = ["patient_id", "site", "y", "wagner_patient_score", "tumor_fraction",
            "artifact_fraction", "attributed_cause", "circumvention_lever"]
    w = ledger[ledger["attributed_cause"].isin(["low-tumor-content", "high-artifact"])]
    return w[[c for c in cols if c in w.columns]]


def _plot_pareto(p, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ov = p[p["scope"] == "overall"].sort_values("count", ascending=False)
    ax.bar(ov["attributed_cause"], ov["count"])
    ax.set_ylabel("errors"); ax.set_title("Wagner error causes (overall)")
    plt.xticks(rotation=30, ha="right"); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():  # pragma: no cover
    print(json.dumps(run_stage_a(), indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
```

- [ ] **Step 4: Run test + real Stage A**

Run: `python -m pytest tests/test_error_anatomy.py -q`
Expected: PASS
Then run the real pass: `python -m argo_deepmsi.eval.error_anatomy` (writes to `results/analysis/error_anatomy/`), and eyeball `A_pareto.csv` + the two worklists.

- [ ] **Step 5: Write E1 doc + stage**

Write `docs/experiments/E1-wagner-error-anatomy-A.md` with the real Pareto table (overall + OAUTHC-first), the per-site decomposition, worklist sizes, and the verified `_MSI_CUTOFF`.

```bash
git add argo_deepmsi/eval/error_anatomy.py tests/test_error_anatomy.py docs/experiments/E1-wagner-error-anatomy-A.md results/analysis/error_anatomy/
python -m pytest tests/test_error_anatomy.py -q
```

---

### Task 4: Stage B — paired-scan natural experiment

**Files:**
- Modify: `argo_deepmsi/eval/error_anatomy.py`
- Test: `tests/test_error_anatomy.py`
- Create (doc, after run): `docs/experiments/E2-wagner-error-anatomy-B.md`

**Interfaces:**
- Consumes: `slide_detail` (Task 1), `cohort_clean.csv`.
- Produces:
  - `paired_scan_analysis(slide_detail: pd.DataFrame) -> tuple[pd.DataFrame, dict]` — pairs each
    retrospective patient's `retrospective_msk` vs `retrospective_oau` slide scores, returns
    `(paired_df, stats)`. `paired_df` cols: `patient_id, p_msk, p_oau, delta_p (oau-msk),
    call_flip (bool at ledger threshold), stain_msk, stain_oau`. `stats` has
    `n_paired, mean_abs_delta, flip_rate, flip_rate_by_stain`.
  - `run_stage_b(threshold, outdir) -> dict` — writes `B_paired_scan.csv`, `B_flip_analysis.png`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_error_anatomy.py  (append)
from argo_deepmsi.eval.error_anatomy import paired_scan_analysis


def test_paired_scan_pairs_only_retrospective_dual():
    detail = pd.DataFrame({
        "slide_id": ["a", "b", "c", "d"],
        "patient_id": ["R1", "R1", "R2", "S1"],
        "p_msih": [0.2, 0.8, 0.3, 0.4],
        "processing_site": ["retrospective_msk", "retrospective_oau",
                             "retrospective_msk", "OAUTHC"],
        "stain_location": ["MSK", "OAU", "MSK", "OAUTHC"],
    })
    paired, stats = paired_scan_analysis(detail)
    # R1 has both msk+oau -> paired; R2 only msk; S1 not retrospective -> excluded
    assert list(paired["patient_id"]) == ["R1"]
    assert abs(paired.loc[0, "delta_p"] - (0.8 - 0.2)) < 1e-9
    assert stats["n_paired"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_error_anatomy.py::test_paired_scan_pairs_only_retrospective_dual -v`
Expected: FAIL (`cannot import name 'paired_scan_analysis'`).

- [ ] **Step 3: Write minimal implementation**

```python
# argo_deepmsi/eval/error_anatomy.py  (append)

def paired_scan_analysis(slide_detail: pd.DataFrame, *, threshold: float | None = None):
    d = slide_detail[slide_detail["processing_site"].isin(
        ["retrospective_msk", "retrospective_oau"])].copy()
    # per (patient, processing_site) collapse to one score (max/sqrt(n) within site)
    def agg(g):
        return float(g["p_msih"].max() / np.sqrt(len(g)))
    site_score = (d.groupby(["patient_id", "processing_site"])
                    .apply(agg).rename("score").reset_index())
    stain = (d.groupby(["patient_id", "processing_site"])["stain_location"]
               .first().reset_index())
    wide = site_score.pivot(index="patient_id", columns="processing_site", values="score")
    wide_stain = stain.pivot(index="patient_id", columns="processing_site", values="stain_location")
    paired = wide.dropna(subset=["retrospective_msk", "retrospective_oau"]).reset_index()
    paired = paired.rename(columns={"retrospective_msk": "p_msk", "retrospective_oau": "p_oau"})
    paired["delta_p"] = paired["p_oau"] - paired["p_msk"]
    paired["stain_msk"] = paired["patient_id"].map(wide_stain["retrospective_msk"])
    paired["stain_oau"] = paired["patient_id"].map(wide_stain["retrospective_oau"])
    if threshold is not None:
        paired["call_flip"] = ((paired["p_msk"] >= threshold) !=
                               (paired["p_oau"] >= threshold))
    else:
        paired["call_flip"] = np.nan
    stats = {
        "n_paired": int(len(paired)),
        "mean_abs_delta": float(paired["delta_p"].abs().mean()) if len(paired) else float("nan"),
        "flip_rate": float(paired["call_flip"].mean()) if threshold is not None and len(paired) else None,
    }
    return paired, stats
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_error_anatomy.py::test_paired_scan_pairs_only_retrospective_dual -v`
Expected: PASS. Then run on real data via a short REPL / add `run_stage_b` and call it, writing `B_paired_scan.csv`.

- [ ] **Step 5: Write E2 doc + stage**

Write `docs/experiments/E2-wagner-error-anatomy-B.md` — the verdict: mean |Δp|, flip rate, whether flips concentrate by stain/imaging site, and how it partitions A's site-structured/unexplained buckets into acquisition-driven vs not. Note that pairs are same-patient, not guaranteed same section.

```bash
git add argo_deepmsi/eval/error_anatomy.py tests/test_error_anatomy.py docs/experiments/E2-wagner-error-anatomy-B.md results/analysis/error_anatomy/
python -m pytest tests/test_error_anatomy.py -q
```

---

### Task 5: Wagner attention-export hook (frozen-equivalence)

**Files:**
- Modify: `argo_deepmsi/models/wagner.py`
- Test: `tests/test_wagner_attention.py`

**Interfaces:**
- Consumes: existing `wagner.py` model classes (the slide `Transformer` / `Attention` with the CLS token).
- Produces:
  - a context manager `capture_attention(model)` that, while active, makes a forward pass also
    record the last layer's CLS→tile attention weights on `model` (attr `last_cls_attn`,
    shape `(n_tiles,)`, summing to ~1 over tiles). Off by default; when not active the forward is
    numerically identical to today.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_wagner_attention.py
import torch
from argo_deepmsi.models import wagner


def test_frozen_equivalence_and_attention_capture():
    torch.manual_seed(0)
    model = wagner.build_slide_transformer_for_test()  # small config helper (add in Task 5)
    x = torch.randn(1, 32, model.tile_dim)             # 32 tiles
    with torch.no_grad():
        p_off = model(x)
    with wagner.capture_attention(model), torch.no_grad():
        p_on = model(x)
    assert torch.allclose(p_off, p_on, atol=1e-6)      # hook must not perturb numerics
    attn = model.last_cls_attn
    assert attn.shape[-1] == 32
    assert abs(float(attn.sum()) - 1.0) < 1e-4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_wagner_attention.py -v`
Expected: FAIL (`build_slide_transformer_for_test` / `capture_attention` missing).

- [ ] **Step 3: Write minimal implementation**

Add to `argo_deepmsi/models/wagner.py`: (a) a tiny `build_slide_transformer_for_test()` factory
that returns the existing slide transformer at a small dim with a `.tile_dim` attribute; (b) a
`capture_attention` context manager that temporarily swaps the last `Attention` layer's
`F.scaled_dot_product_attention` call for an explicit path that stores CLS→tile weights.

```python
# argo_deepmsi/models/wagner.py  (additions)
import contextlib
import torch.nn.functional as F


@contextlib.contextmanager
def capture_attention(model):
    """Temporarily record last-layer CLS->tile attention on `model.last_cls_attn`.

    Additive and numerically inert: the stored weights come from an explicit
    softmax(QK^T/sqrt(d)) computed alongside the normal fused SDPA output, which is
    what the frozen forward already returns.
    """
    target = _last_attention_module(model)
    orig = target.forward

    def patched(x):
        q, k, v = target.to_qkv(x)  # match the module's real projection API
        d = q.shape[-1]
        attn = torch.softmax(q @ k.transpose(-2, -1) / (d ** 0.5), dim=-1)
        model.last_cls_attn = attn[..., 0, 1:].mean(dim=tuple(range(attn.ndim - 2))).detach()
        return orig(x)  # return the untouched fused-SDPA output for numerics

    target.forward = patched
    try:
        yield
    finally:
        target.forward = orig
```

**Note for implementer:** the exact QKV extraction (`target.to_qkv`, head reshape) must match the
real `Attention` module in `wagner.py:84`. Read that class first and adapt the patched path so the
recomputed `attn` reflects the same projection/heads; the frozen-equivalence assertion is your
guardrail — `orig(x)` is returned so `p_msih` never changes.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_wagner_attention.py -v`
Expected: PASS

- [ ] **Step 5: Stage + gate**

```bash
git add argo_deepmsi/models/wagner.py tests/test_wagner_attention.py
python -m pytest tests/test_wagner_attention.py -q
```

---

### Task 6: Stage C — attention-mass on tumor + overlays + SLURM wrapper

**Files:**
- Modify: `argo_deepmsi/eval/error_anatomy.py`
- Create: `scripts/error_anatomy_c.sh`
- Test: `tests/test_error_anatomy.py`

**Interfaces:**
- Consumes: `capture_attention` (Task 5), `results/data/tumor_tiles/<slide>.npy` masks, the
  A/B ledger (to pick the confident-wrong + unexplained/site-structured subset).
- Produces:
  - `attention_mass_on_tumor(attn: np.ndarray, tumor_idx: np.ndarray, n_tiles: int) -> float` —
    fraction of attention mass on tumor tiles.
  - `run_stage_c(ledger_path, outdir, device="cuda") -> pd.DataFrame` — re-infers Wagner with
    attention capture on the error subset, computes tumor attention mass vs matched correct
    controls, writes `C_attention_summary.csv` and per-slide overlay PNGs.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_error_anatomy.py  (append)
import numpy as np
from argo_deepmsi.eval.error_anatomy import attention_mass_on_tumor


def test_attention_mass_on_tumor():
    attn = np.array([0.1, 0.6, 0.2, 0.1])   # sums to 1
    tumor_idx = np.array([1, 2])            # tiles 1 and 2 are tumor
    assert abs(attention_mass_on_tumor(attn, tumor_idx, n_tiles=4) - 0.8) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_error_anatomy.py::test_attention_mass_on_tumor -v`
Expected: FAIL (`cannot import name 'attention_mass_on_tumor'`).

- [ ] **Step 3: Write minimal implementation + SLURM wrapper**

```python
# argo_deepmsi/eval/error_anatomy.py  (append)

def attention_mass_on_tumor(attn, tumor_idx, n_tiles) -> float:
    a = np.asarray(attn, dtype=float)
    a = a / a.sum() if a.sum() else a
    mask = np.zeros(n_tiles, dtype=bool)
    mask[np.asarray(tumor_idx, dtype=int)] = True
    return float(a[mask].sum())
```

`run_stage_c` (write in this step, no unit test — exercised on real GPU) selects the subset from
the ledger (`confident_wrong & attributed_cause in {"unexplained","site-structured"}`) plus a
matched set of correct controls, reopens each slide's cached zarr
(`open_wsi(svs, store=svs.parent, attach_thumbnail=False)` per the CLAUDE.md zarr-reopen rule),
runs Wagner under `capture_attention`, computes `attention_mass_on_tumor`, and saves overlays.

```bash
# scripts/error_anatomy_c.sh
#!/bin/bash
#SBATCH --job-name=mdiberna_erranatomy_c
#SBATCH --output=scripts/logs/erranatomy_c_%j.out
#SBATCH --time=02:00:00
#SBATCH --partition=nvidia-A6000-20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail
source /lab/barcheese01/mdiberna/miniconda3/etc/profile.d/conda.sh
conda activate argo
cd /lab/barcheese01/mdiberna/ARGO-DeepMSI
python -c "from argo_deepmsi.eval.error_anatomy import run_stage_c; \
  run_stage_c('results/analysis/error_anatomy/A_error_ledger.csv', \
              'results/analysis/error_anatomy')"
```

- [ ] **Step 4: Run unit test + submit GPU pass**

Run: `python -m pytest tests/test_error_anatomy.py::test_attention_mass_on_tumor -v` → PASS.
Then: `sbatch scripts/error_anatomy_c.sh` and monitor `scripts/logs/erranatomy_c_*.out`.

- [ ] **Step 5: Stage + gate**

```bash
git add argo_deepmsi/eval/error_anatomy.py scripts/error_anatomy_c.sh tests/test_error_anatomy.py results/analysis/error_anatomy/
python -m pytest tests/test_error_anatomy.py -q
```

---

### Task 7: Synthesis — the recipe

**Files:**
- Modify: `argo_deepmsi/eval/error_anatomy.py`, `docs/summary.md`
- Create: `docs/experiments/E4-error-anatomy-synthesis.md`
- Test: `tests/test_error_anatomy.py`

**Interfaces:**
- Consumes: A ledger + Pareto, B stats, C summary.
- Produces:
  - `synthesize(ledger, b_stats, c_summary=None) -> pd.DataFrame` — one row per cause with
    `count, pct_of_errors, circumvention_lever`; `pct_of_errors` sums to 100 over error rows.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_error_anatomy.py  (append)
from argo_deepmsi.eval.error_anatomy import synthesize


def test_synthesis_percentages_sum_to_100():
    ledger = pd.DataFrame({
        "attributed_cause": ["label-suspect", "label-suspect", "high-artifact", "correct"],
        "circumvention_lever": ["x", "x", "y", ""],
    })
    s = synthesize(ledger, b_stats={"flip_rate": 0.3})
    assert abs(s["pct_of_errors"].sum() - 100.0) < 1e-6
    assert set(s["attributed_cause"]) == {"label-suspect", "high-artifact"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_error_anatomy.py::test_synthesis_percentages_sum_to_100 -v`
Expected: FAIL (`cannot import name 'synthesize'`).

- [ ] **Step 3: Write minimal implementation**

```python
# argo_deepmsi/eval/error_anatomy.py  (append)

def synthesize(ledger, b_stats=None, c_summary=None) -> pd.DataFrame:
    err = ledger[ledger["attributed_cause"] != "correct"]
    n = len(err)
    g = (err.groupby("attributed_cause")
            .agg(count=("attributed_cause", "size"),
                 circumvention_lever=("circumvention_lever", "first"))
            .reset_index())
    g["pct_of_errors"] = 100.0 * g["count"] / n if n else 0.0
    return g.sort_values("count", ascending=False).reset_index(drop=True)
```

- [ ] **Step 4: Run test + real synthesis**

Run: `python -m pytest tests/test_error_anatomy.py -q` → PASS.
Then produce the real synthesis table + figure and write `docs/experiments/E4-error-anatomy-synthesis.md`; add a short "Error anatomy → next levers" section to `docs/summary.md` replacing "Current path forward".

- [ ] **Step 5: Final gate + stage**

```bash
git add argo_deepmsi/eval/error_anatomy.py docs/experiments/E4-error-anatomy-synthesis.md docs/summary.md tests/test_error_anatomy.py results/analysis/error_anatomy/
python -m pytest tests/test_error_anatomy.py tests/test_wagner_attention.py -q
bash ralph/verify.sh
```

---

## Self-review notes

- **Spec coverage:** ledger (T1), attribution+levers (T2), Pareto/worklists/per-site (T3),
  paired-scan (T4), attention hook w/ frozen-equivalence (T5), attention-mass + overlays + SLURM
  (T6), synthesis + summary fold-in (T7). All spec sections covered.
- **Reuse:** aggregation and screening helpers are imported, not re-implemented.
- **No-commit rule:** every task ends in `git add` + gate, never `git commit`.
- **Open item for implementer:** verify `_MSI_CUTOFF` against `data_ingestion.py`; adapt the
  `capture_attention` QKV path to the real `Attention` module (frozen-equivalence test guards it).
