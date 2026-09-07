"""Wagner error anatomy — ledger, attribution, paired-scan, synthesis.

Diagnoses *which* patients/slides the frozen Wagner champion gets wrong and *why*,
then maps each cause to a circumvention lever. Pure functions over dataframes; I/O
lives in ``run_stage_a`` / ``run_stage_b`` / ``run_stage_c``. See
``docs/superpowers/specs/2026-08-04-wagner-error-anatomy-design.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from argo_deepmsi.eval.screening import threshold_at_sensitivity

# --- attribution constants (recorded in the manifest) ---
TUMOR_FLOOR = 0.01
ARTIFACT_CEILING = 0.5
CMO_SCORE_BAND = 0.10
CONFIDENT_MARGIN = 0.15
# cmo_msi_score is MSIsensor-style %; cmo_msi_status is Stable (<3), Indeterminate (3-10),
# Instable (>=10). Label = isMSIH from cmo_msi_status, so the MSI-H cutoff is 10.
_MSI_CUTOFF = 10.0


def build_error_ledger(scores_df, cohort_df, clinical_df, *, target_sensitivity=0.95,
                       threshold=None):
    """Join Wagner slide scores + covariates into a patient ledger and slide detail.

    Returns ``(patient_ledger, slide_detail)``. If ``threshold`` is None it is derived
    from ``target_sensitivity``; otherwise the given threshold is used (Stages B/C reuse it).
    """
    s = scores_df.copy()
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
        default="",
    )
    agg["rank_residual"] = agg["wagner_patient_score"] - thr
    agg["confident_wrong"] = (
        ((agg["error_class"] == "FP") & (agg["rank_residual"] > CONFIDENT_MARGIN))
        | ((agg["error_class"] == "FN") & (agg["rank_residual"] < -CONFIDENT_MARGIN))
    )

    clin = clinical_df.rename(columns={
        "PATIENT": "patient_id",
        "slide_imaging_site": "imaging_site",
    })
    keep_clin = [c for c in ["patient_id", "msi_method", "imaging_site"] if c in clin.columns]
    cov = _patient_covariates(cohort_df)
    ledger = (agg.merge(cov, on="patient_id", how="left", suffixes=("", "_cov"))
                 .merge(clin[keep_clin], on="patient_id", how="left"))
    ledger.attrs["threshold"] = float(thr)

    # slide-level detail: QC keyed by slide when the cohort is slide-level, else by patient
    qc = ["n_tiles", "tumor_fraction", "artifact_fraction", "processing_site"]
    if "slide_id" in cohort_df.columns:
        key, keep = "slide_id", ["slide_id"] + [c for c in qc if c in cohort_df.columns]
        detail = s.merge(cohort_df[keep], on=key, how="left")
    else:
        key, keep = "patient_id", ["patient_id"] + [c for c in qc if c in cohort_df.columns]
        detail = s.merge(cohort_df[keep].drop_duplicates(key), on=key, how="left")
    return ledger, detail


def _patient_covariates(cohort_df):
    """Collapse the slide-level cohort table to one covariate row per patient.

    Label attributes are constant within a patient (first); QC fractions are averaged and
    tile counts summed across the patient's slides.
    """
    df = cohort_df.copy()
    df["mmr_cmo_concordance"] = _mmr_cmo_concordance(df).to_numpy()
    first_cols = [c for c in ["processing_site", "patient_cohort", "cmo_msi_status",
                              "cmo_msi_score", "msi_status_mmr", "label_source",
                              "label_certainty", "mmr_cmo_concordance"] if c in df.columns]
    mean_cols = [c for c in ["tumor_fraction", "artifact_fraction"] if c in df.columns]
    sum_cols = [c for c in ["n_tiles", "n_tumor_tiles"] if c in df.columns]
    spec = {**{c: "first" for c in first_cols},
            **{c: "mean" for c in mean_cols},
            **{c: "sum" for c in sum_cols}}
    return df.groupby("patient_id").agg(spec).reset_index()


def _mmr_cmo_concordance(cohort_df):
    """True where CMO status and MMR status disagree on MSI-H (a label-suspect signal)."""
    def norm(v):
        v = str(v).upper()
        if "MSS" in v:
            return "MSS"
        if "MSI" in v:
            return "MSIH"
        return None
    cmo = cohort_df["cmo_msi_status"].map(norm) if "cmo_msi_status" in cohort_df else pd.Series([None] * len(cohort_df))
    mmr = cohort_df["msi_status_mmr"].map(norm) if "msi_status_mmr" in cohort_df else pd.Series([None] * len(cohort_df))
    return ~((cmo == mmr) | cmo.isna() | mmr.isna())


_LEVERS = {
    "label-suspect": "re-adjudicate label (molecular/pathology worklist); label-certainty re-run",
    "low-tumor-content": "re-tile / stronger tumor detection (QC-refix worklist)",
    "high-artifact": "artifact removal / re-scan (QC-refix worklist)",
    "borderline-score": "operating-point / selective abstention choice",
    "site-structured": "Stage B: acquisition vs biology",
    "unexplained": "Stage C: spatial attention introspection",
    "correct": "",
}


# covariate causes in priority order; each maps to a boolean predicate over the ledger
_COVARIATE_PRIORITY = ["label-suspect", "low-tumor-content", "high-artifact"]


def _cause_predicates(df: pd.DataFrame) -> dict:
    """Boolean Series per covariate cause, aligned to df.index."""
    score = pd.to_numeric(df.get("cmo_msi_score"), errors="coerce")
    near_cutoff = (score - _MSI_CUTOFF).abs() <= CMO_SCORE_BAND * _MSI_CUTOFF
    label_certainty = df.get("label_certainty", pd.Series(index=df.index, dtype=object))
    concord = df.get("mmr_cmo_concordance", pd.Series(False, index=df.index)).eq(True)
    label_suspect = (label_certainty.eq("indeterminate_as_mss") | concord | near_cutoff.fillna(False))
    low_tumor = pd.to_numeric(df.get("tumor_fraction"), errors="coerce").le(TUMOR_FLOOR).fillna(False)
    high_artifact = pd.to_numeric(df.get("artifact_fraction"), errors="coerce").ge(ARTIFACT_CEILING).fillna(False)
    return {"label-suspect": label_suspect.fillna(False),
            "low-tumor-content": low_tumor,
            "high-artifact": high_artifact}


def compute_enrichment(ledger: pd.DataFrame, preds: dict) -> dict:
    """P(cond|error) / P(cond|correct) per covariate cause (the causal-signal guard)."""
    err = ledger["error_class"].isin(["FP", "FN"])
    correct = ledger["error_class"].isin(["TP", "TN"])
    stats = {}
    for c, p in preds.items():
        pe = float(p[err].mean()) if err.any() else 0.0
        pc = float(p[correct].mean()) if correct.any() else 0.0
        ratio = (pe / pc) if pc > 0 else (float("inf") if pe > 0 else 0.0)
        stats[c] = {"p_err": pe, "p_correct": pc, "enrichment": ratio}
    return stats


def attribute_causes(ledger: pd.DataFrame, *, enrichment_min: float = 1.2) -> pd.DataFrame:
    """Attribute each error to a cause, but only to a covariate that is *enriched* among errors.

    A covariate bucket (e.g. high-artifact) may claim an error only if that covariate is
    over-represented in errors vs correct predictions by >= ``enrichment_min``. Ubiquitous but
    non-causal covariates fall through to ``unexplained`` instead of dominating the Pareto.
    """
    out = ledger.reset_index(drop=True).copy()
    preds = _cause_predicates(out)
    enr = compute_enrichment(out, preds)
    enriched = {c for c, s in enr.items() if s["enrichment"] >= enrichment_min}

    causes = []
    for i in out.index:
        if out.at[i, "error_class"] not in ("FP", "FN"):
            causes.append("correct")
            continue
        assigned = next((c for c in _COVARIATE_PRIORITY if c in enriched and bool(preds[c].at[i])), None)
        if assigned is None:
            assigned = "unexplained" if bool(out.at[i, "confident_wrong"]) else "borderline-score"
        causes.append(assigned)
    out["attributed_cause"] = causes
    out["secondary_cause"] = ""
    out["circumvention_lever"] = out["attributed_cause"].map(_LEVERS).fillna("")
    out.attrs["enrichment"] = enr
    out.attrs["enriched_causes"] = sorted(enriched)
    return out


def assign_site_structured(ledger: pd.DataFrame, *, margin: float = 0.10) -> pd.DataFrame:
    """Reclassify `unexplained` errors to `site-structured` in sites with elevated error rate."""
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
    """Cause x count for error rows, overall and OAUTHC-first."""
    err = ledger[ledger["attributed_cause"] != "correct"]

    def _counts(sub, scope):
        c = (sub["attributed_cause"].value_counts()
             .rename_axis("attributed_cause").reset_index(name="count"))
        c["scope"] = scope
        return c

    overall = _counts(err, "overall")
    oau = _counts(err[err["site"] == "OAUTHC"], "OAUTHC")
    return pd.concat([overall, oau], ignore_index=True)


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
    ax.set_ylabel("errors")
    ax.set_title("Wagner error causes (overall)")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_stage_a(outdir: str = "results/analysis/error_anatomy") -> dict:
    """Build + attribute the ledger from cached artifacts; write all Stage-A outputs."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv("results/scorers/wagner_zeroshot/slide_scores.csv")
    cohort = pd.read_csv("results/data/cohort_clean.csv")
    clinical = pd.read_csv("results/data/clinical_table.csv")
    ledger, detail = build_error_ledger(scores, cohort, clinical)
    thr = ledger.attrs.get("threshold")
    ledger = assign_site_structured(attribute_causes(ledger))
    ledger.to_csv(out / "A_error_ledger.csv", index=False)
    detail.to_csv(out / "A_slide_detail.csv", index=False)
    p = pareto(ledger)
    p.to_csv(out / "A_pareto.csv", index=False)
    _plot_pareto(p, out / "A_pareto.png")
    _per_site(ledger).to_csv(out / "A_per_site_decomposition.csv", index=False)
    _worklist_label(ledger).to_csv(out / "worklist_label_adjudication.csv", index=False)
    _worklist_qc(ledger).to_csv(out / "worklist_qc_refix.csv", index=False)
    manifest = {
        "threshold": thr,
        "msi_cutoff": _MSI_CUTOFF,
        "tumor_floor": TUMOR_FLOOR,
        "artifact_ceiling": ARTIFACT_CEILING,
        "cmo_score_band": CMO_SCORE_BAND,
        "confident_margin": CONFIDENT_MARGIN,
        "n_patients": int(len(ledger)),
        "n_errors": int((ledger["attributed_cause"] != "correct").sum()),
    }
    (out / "A_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():  # pragma: no cover
    print(json.dumps(run_stage_a(), indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()


def paired_scan_analysis(slide_detail: pd.DataFrame, *, threshold: float | None = None):
    """Compare Wagner scores across the MSK vs OAU acquisition of the same retrospective tissue.

    Returns ``(paired_df, stats)``. Large within-patient |Δp| on identical biology indicates an
    acquisition shortcut; small Δp with a persistent error points to biology/label.
    """
    d = slide_detail[slide_detail["processing_site"].isin(
        ["retrospective_msk", "retrospective_oau"])].copy()

    site_score = (
        d.groupby(["patient_id", "processing_site"])
        .agg(_max=("p_msih", "max"), _n=("p_msih", "size"))
        .assign(score=lambda frame: frame["_max"] / np.sqrt(frame["_n"]))
        .drop(columns=["_max", "_n"])
        .reset_index()
    )
    stain = (d.groupby(["patient_id", "processing_site"])["stain_location"]
               .first().reset_index())
    wide = site_score.pivot(index="patient_id", columns="processing_site", values="score")
    wide_stain = stain.pivot(index="patient_id", columns="processing_site", values="stain_location")
    need = ["retrospective_msk", "retrospective_oau"]
    for col in need:
        if col not in wide.columns:
            wide[col] = np.nan
        if col not in wide_stain.columns:
            wide_stain[col] = np.nan
    paired = wide.dropna(subset=need).reset_index()
    paired = paired.rename(columns={"retrospective_msk": "p_msk", "retrospective_oau": "p_oau"})
    paired["delta_p"] = paired["p_oau"] - paired["p_msk"]
    paired["stain_msk"] = paired["patient_id"].map(wide_stain["retrospective_msk"])
    paired["stain_oau"] = paired["patient_id"].map(wide_stain["retrospective_oau"])
    if threshold is not None:
        paired["call_flip"] = ((paired["p_msk"] >= threshold) != (paired["p_oau"] >= threshold))
    else:
        paired["call_flip"] = np.nan
    stats = {
        "n_paired": int(len(paired)),
        "mean_abs_delta": float(paired["delta_p"].abs().mean()) if len(paired) else float("nan"),
        "flip_rate": (float(paired["call_flip"].mean())
                      if threshold is not None and len(paired) else None),
    }
    return paired, stats


def run_stage_b(threshold: float, outdir: str = "results/analysis/error_anatomy") -> dict:
    """Run the paired-scan experiment on the cached cohort; write B_paired_scan.csv + figure."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv("results/scorers/wagner_zeroshot/slide_scores.csv")
    cohort = pd.read_csv("results/data/cohort_clean.csv")
    clinical = pd.read_csv("results/data/clinical_table.csv")
    _, detail = build_error_ledger(scores, cohort, clinical, threshold=threshold)
    paired, stats = paired_scan_analysis(detail, threshold=threshold)
    paired.to_csv(out / "B_paired_scan.csv", index=False)
    _plot_paired(paired, out / "B_flip_analysis.png")
    (out / "B_manifest.json").write_text(json.dumps(stats, indent=2))
    return stats


def _plot_paired(paired, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    if len(paired):
        ax.scatter(paired["p_msk"], paired["p_oau"], s=18, alpha=0.7)
        lim = [0, max(paired[["p_msk", "p_oau"]].max().max(), 0.1)]
        ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("Wagner p_msih — MSK scan")
    ax.set_ylabel("Wagner p_msih — OAU scan")
    ax.set_title("Paired retrospective scans (same tissue)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def attention_mass_on_tumor(attn, tumor_idx, n_tiles) -> float:
    """Fraction of (renormalised) CLS attention mass falling on tumor tiles."""
    a = np.asarray(attn, dtype=float)
    total = a.sum()
    a = a / total if total else a
    mask = np.zeros(int(n_tiles), dtype=bool)
    idx = np.asarray(tumor_idx, dtype=int)
    idx = idx[(idx >= 0) & (idx < n_tiles)]
    mask[idx] = True
    return float(a[mask].sum())


def _stage_c_subset(ledger: pd.DataFrame, *, n_controls_per_error: int = 1) -> pd.DataFrame:
    """Confident, still-unexplained error patients + matched correct (TN) controls."""
    err = ledger[(ledger["attributed_cause"].isin(["unexplained", "site-structured"]))
                 & (ledger["error_class"].isin(["FP", "FN"]))].copy()
    err["group"] = "error"
    n_ctrl = min(len(ledger[ledger["error_class"] == "TN"]),
                 n_controls_per_error * len(err))
    ctrl = (ledger[ledger["error_class"] == "TN"]
            .sort_values("wagner_patient_score", ascending=False).head(n_ctrl).copy())
    ctrl["group"] = "control"
    return pd.concat([err, ctrl], ignore_index=True)


def run_stage_c(ledger_path: str = "results/analysis/error_anatomy/A_error_ledger.csv",
                outdir: str = "results/analysis/error_anatomy", *, device: str = "cpu") -> pd.DataFrame:
    """Score the error/control subset with Wagner attention capture; measure tumor attention mass.

    Wagner runs on cached CTransPath tile features, so this is CPU-cheap and needs no GPU.
    """
    import torch
    from wsidata import open_wsi

    from argo_deepmsi.models.wagner import capture_attention, load_wagner

    out = Path(outdir)
    ledger = pd.read_csv(ledger_path)
    detail = pd.read_csv(out / "A_slide_detail.csv")
    subset = _stage_c_subset(ledger)
    groups = dict(zip(subset["patient_id"], subset["group"]))
    slides = detail[detail["patient_id"].isin(subset["patient_id"])].copy()

    stable = pd.read_csv("results/data/slide_table_pyramidal.csv")
    stable["slide_id"] = stable["FILENAME"].map(lambda p: Path(p).stem)
    fname = dict(zip(stable["slide_id"], stable["FILENAME"]))

    tumor_dir = Path("results/data/tumor_tiles")
    model = load_wagner(device=device)
    rows = []
    with torch.no_grad():
        for _, r in slides.iterrows():
            sid = r["slide_id"]
            fpath = fname.get(sid)
            tmask = tumor_dir / f"{sid}.npy"
            if fpath is None or not tmask.exists():
                continue
            svs = Path(fpath)
            try:
                wsi = open_wsi(str(svs), store=str(svs.parent), attach_thumbnail=False)
                if "ctranspath_tiles" not in wsi.tables:
                    continue
                X = np.asarray(wsi.tables["ctranspath_tiles"].X, dtype=np.float32)
                if X.shape[0] == 0:
                    continue
                x = torch.from_numpy(X).unsqueeze(0).to(device)
                with capture_attention(model):
                    _ = model(x)
                attn = model.last_cls_attn.cpu().numpy()
                tumor_idx = np.load(tmask)
                mass = attention_mass_on_tumor(attn, tumor_idx, n_tiles=X.shape[0])
                rows.append({"slide_id": sid, "patient_id": r["patient_id"],
                             "group": groups.get(r["patient_id"]), "n_tiles": int(X.shape[0]),
                             "n_tumor_tiles": int(len(tumor_idx)),
                             "tumor_attention_mass": mass,
                             "tumor_tile_fraction": len(tumor_idx) / X.shape[0]})
            except Exception as e:  # transient zarr / read issue -> skip, re-runnable
                print(f"  skip {sid}: {e}")
                continue

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "C_attention_summary.csv", index=False)
    if len(summary):
        agg = summary.groupby("group").agg(
            n=("slide_id", "size"),
            mean_tumor_attention=("tumor_attention_mass", "mean"),
            mean_tumor_fraction=("tumor_tile_fraction", "mean"),
        ).reset_index()
        (out / "C_manifest.json").write_text(
            json.dumps(agg.to_dict(orient="records"), indent=2))
        print(agg.to_string(index=False))
    return summary


def synthesize(ledger, b_stats=None, c_summary=None) -> pd.DataFrame:
    """One row per error cause with count, pct of errors, and its circumvention lever."""
    err = ledger[ledger["attributed_cause"] != "correct"]
    n = len(err)
    g = (err.groupby("attributed_cause")
            .agg(count=("attributed_cause", "size"),
                 circumvention_lever=("circumvention_lever", "first"))
            .reset_index())
    g["pct_of_errors"] = 100.0 * g["count"] / n if n else 0.0
    return g.sort_values("count", ascending=False).reset_index(drop=True)
