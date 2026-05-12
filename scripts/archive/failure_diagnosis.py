"""Step 1 — Failure diagnosis on the Wagner zero-shot classifier.

Covers 1a (tissue-area stratification), 1c (error profiling by metadata
covariates), 1d (OAUTHC prospective deep dive) from docs/remaining_tasks.md.

1b (MSIsensor correlation) is blocked on missing IMPACT data and is skipped.

Inputs (all already on disk):
    results/analysis/wagner_zeroshot/slide_scores.csv
    results/analysis/wagner_zeroshot/patient_scores.csv
    results/data/slide_table.csv          (stain/cut/image location)
    results/data/clinical_table_full.csv  (batch_number → label derivation)

Outputs:
    results/analysis/failure_diagnosis/
        slide_stats.csv       enriched per-slide table
        tercile_auroc.csv     per-tercile Wagner AUROC, overall + per-site
        error_coefs.csv       logit(correct ~ covariates) coefficients
        oauthc_audit.md       prospective-vs-retro patient overlap
        tissue_vs_p.png       scatter: n_tiles vs Wagner p, coloured by label
        tercile_auroc.png     per-tercile AUROC bar
        error_coefs.png       coefficient bar

Usage:
    python scripts/failure_diagnosis.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

TILE_AREA_UM2 = 256 * 256 * 0.5 * 0.5  # tile_px=256, mpp=0.5


def _safe_auroc(y, p):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _safe_auprc(y, p):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def load_enriched(wagner_dir: Path, slide_table: Path, clinical_full: Path) -> pd.DataFrame:
    ss = pd.read_csv(wagner_dir / "slide_scores.csv")
    st = pd.read_csv(slide_table)
    st["slide_id"] = st["FILENAME"].apply(lambda p: Path(p).stem)
    st = st[["slide_id", "PATIENT", "SITE", "cut_location", "stain_location",
             "image_location"]].rename(columns={"PATIENT": "patient_id", "SITE": "site_full"})

    cl = pd.read_csv(clinical_full)[["PATIENT", "batch_number", "redcap_data_access_group"]]
    cl = cl.rename(columns={"PATIENT": "patient_id"})
    cl["batch_number"] = cl["batch_number"].astype(str)
    cl["label_source"] = np.where(cl["batch_number"].isin(["1", "2"]),
                                  "retrospective_mmr", "prospective_cmo")

    df = ss.merge(st, on="slide_id", how="left", suffixes=("", "_st"))
    df = df.merge(cl, on="patient_id", how="left")
    df["tissue_area_um2"] = df["n_tiles"].astype(float) * TILE_AREA_UM2
    df["tissue_area_mm2"] = df["tissue_area_um2"] / 1e6
    df["pred"] = (df["p_msih"] >= 0.5).astype(int)
    df["correct"] = (df["pred"] == df["y"]).astype(int)
    return df


# ---------------------------------------------------------------- 1a ------

def tercile_analysis(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows = []
    df = df.copy()
    df["tercile"] = pd.qcut(df["n_tiles"], q=3, labels=["small", "medium", "large"])
    groups = [("overall", df)]
    for site, sub in df.groupby("site"):
        groups.append((site, sub))
    for name, sub in groups:
        for ter, s2 in sub.groupby("tercile", observed=True):
            rows.append(dict(
                stratum=name, tercile=str(ter),
                n=len(s2), prevalence=float(s2["y"].mean()),
                tile_min=int(s2["n_tiles"].min()), tile_max=int(s2["n_tiles"].max()),
                tile_median=float(s2["n_tiles"].median()),
                auroc=_safe_auroc(s2["y"], s2["p_msih"]),
                auprc=_safe_auprc(s2["y"], s2["p_msih"]),
            ))
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "tercile_auroc.csv", index=False)

    # Bar plot (overall + per-site, side by side)
    piv = (out.pivot(index="stratum", columns="tercile", values="auroc")
              .reindex(columns=["small", "medium", "large"]))
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(piv)), 4), dpi=150)
    x = np.arange(len(piv))
    w = 0.27
    for i, col in enumerate(piv.columns):
        ax.bar(x + (i - 1) * w, piv[col].values, width=w, label=col)
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index, rotation=35, ha="right")
    ax.set_ylabel("Wagner AUROC")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, ls="--", c="grey", lw=0.8)
    ax.legend(title="tissue-area tercile", fontsize=8)
    ax.set_title("1a — Wagner AUROC by tissue-area tercile (n_tiles proxy)")
    fig.tight_layout()
    fig.savefig(outdir / "tercile_auroc.png", bbox_inches="tight")
    plt.close(fig)

    # Scatter
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    for y_val, colour, lbl in [(0, "#4477AA", "MSS"), (1, "#CC3311", "MSI-H")]:
        s = df[df["y"] == y_val]
        ax.scatter(s["n_tiles"], s["p_msih"], s=10, alpha=0.5,
                   c=colour, label=f"{lbl} (n={len(s)})", edgecolors="none")
    ax.set_xscale("log")
    ax.axhline(0.5, ls="--", c="grey", lw=0.8)
    ax.set_xlabel("n_tiles per slide (log)")
    ax.set_ylabel("Wagner P(MSI-H)")
    ax.set_title("1a — Wagner probability vs tissue area")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "tissue_vs_p.png", bbox_inches="tight")
    plt.close(fig)

    return out


# ---------------------------------------------------------------- 1c ------

def error_profiling(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    """Logistic regression of correct-prediction on metadata covariates."""
    feat = df[["site", "stain_location", "cut_location", "label_source"]].astype(str).fillna("NA")
    X_cat = pd.get_dummies(feat, drop_first=True)
    # Numeric: log10(n_tiles), y (prevalence effect)
    X_num = pd.DataFrame({
        "log10_n_tiles": np.log10(df["n_tiles"].clip(lower=1).values),
        "y": df["y"].values.astype(float),
    })
    X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    scaler = StandardScaler()
    Xs = X.copy()
    Xs[["log10_n_tiles"]] = scaler.fit_transform(X[["log10_n_tiles"]].values)
    y = df["correct"].astype(int).values

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, random_state=42)
    clf.fit(Xs, y)
    coefs = pd.DataFrame({"feature": X.columns, "coef": clf.coef_[0]})
    coefs["abs"] = coefs["coef"].abs()
    coefs = coefs.sort_values("abs", ascending=False).drop(columns="abs")
    coefs.to_csv(outdir / "error_coefs.csv", index=False)

    # Per-site confusion matrix-ish summary
    per_site = (df.groupby("site").agg(
        n=("slide_id", "size"),
        prevalence=("y", "mean"),
        pred_rate=("pred", "mean"),
        accuracy=("correct", "mean"),
        auroc=("p_msih", lambda s: _safe_auroc(df.loc[s.index, "y"], s)),
    ).reset_index())
    per_site.to_csv(outdir / "per_site_errors.csv", index=False)

    # Coefficient bar
    top = coefs.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(top) + 1.5), dpi=150)
    colours = ["#CC3311" if c < 0 else "#228833" for c in top["coef"]]
    ax.barh(top["feature"], top["coef"], color=colours)
    ax.axvline(0, c="k", lw=0.6)
    ax.set_xlabel("logit(correct) coefficient")
    ax.set_title("1c — Covariates driving Wagner prediction errors\n(+ = helps accuracy, − = hurts)")
    fig.tight_layout()
    fig.savefig(outdir / "error_coefs.png", bbox_inches="tight")
    plt.close(fig)
    return coefs


# ---------------------------------------------------------------- 1d ------

def oauthc_audit(df: pd.DataFrame, outdir: Path) -> None:
    lines = ["# 1d — OAUTHC Prospective Deep Dive\n"]

    oauthc_pros = df[(df["site"] == "OAUTHC")]
    oauthc_retro = df[(df["site"] == "retrospective_oau")]

    lines.append(f"- OAUTHC (prospective): {len(oauthc_pros)} slides, "
                 f"{oauthc_pros['patient_id'].nunique()} patients, "
                 f"prevalence {oauthc_pros['y'].mean():.3f}, "
                 f"Wagner AUROC {_safe_auroc(oauthc_pros['y'], oauthc_pros['p_msih']):.3f}.")
    lines.append(f"- retrospective_oau: {len(oauthc_retro)} slides, "
                 f"{oauthc_retro['patient_id'].nunique()} patients, "
                 f"prevalence {oauthc_retro['y'].mean():.3f}, "
                 f"Wagner AUROC {_safe_auroc(oauthc_retro['y'], oauthc_retro['p_msih']):.3f}.\n")

    # Patient overlap
    p_pros = set(oauthc_pros["patient_id"])
    p_retro = set(oauthc_retro["patient_id"])
    shared = p_pros & p_retro
    lines.append(f"## Patient overlap\n")
    lines.append(f"- Patients with slides in BOTH OAUTHC prospective and retrospective_oau: **{len(shared)}**")
    lines.append(f"- Prospective-only patients: {len(p_pros - p_retro)}")
    lines.append(f"- Retro-only patients: {len(p_retro - p_pros)}\n")

    # Tissue area & tile count
    def _stats(sub, name):
        return (f"- **{name}** n_tiles: median={sub['n_tiles'].median():.0f}  "
                f"p25={sub['n_tiles'].quantile(0.25):.0f}  "
                f"p75={sub['n_tiles'].quantile(0.75):.0f}  "
                f"min={sub['n_tiles'].min()}  max={sub['n_tiles'].max()}")
    lines.append("## Tissue area (n_tiles as proxy)\n")
    lines.append(_stats(oauthc_pros, "OAUTHC prospective"))
    lines.append(_stats(oauthc_retro, "retrospective_oau"))
    for site in sorted(df["site"].unique()):
        if site in ("OAUTHC", "retrospective_oau"):
            continue
        lines.append(_stats(df[df["site"] == site], site))
    lines.append("")

    # Label source breakdown per site
    lines.append("## Label derivation source by site\n")
    lines.append("| site | prospective_cmo | retrospective_mmr | NA |")
    lines.append("|---|---|---|---|")
    for site, sub in df.groupby("site"):
        counts = sub["label_source"].value_counts(dropna=False)
        lines.append(f"| {site} | {int(counts.get('prospective_cmo', 0))} | "
                     f"{int(counts.get('retrospective_mmr', 0))} | "
                     f"{int(sub['label_source'].isna().sum())} |")
    lines.append("")

    # Concordance where both exist
    if shared:
        lines.append("## MSI label concordance for the overlap patients\n")
        rows = []
        for pid in sorted(shared):
            lab_pros = oauthc_pros.loc[oauthc_pros["patient_id"] == pid, "y"].iloc[0]
            lab_retro = oauthc_retro.loc[oauthc_retro["patient_id"] == pid, "y"].iloc[0]
            rows.append((pid, lab_pros, lab_retro, lab_pros == lab_retro))
        concordance = sum(1 for r in rows if r[3]) / len(rows)
        lines.append(f"- {len(rows)} overlap patients, label concordance: **{concordance:.3f}**")
        discordant = [r for r in rows if not r[3]]
        if discordant:
            lines.append("- Discordant patients:")
            for pid, lp, lr, _ in discordant:
                lines.append(f"  - {pid}: prospective y={lp}, retro y={lr}")
        lines.append("")
    else:
        lines.append("No patient overlap between OAUTHC prospective and retrospective_oau; "
                     "label-concordance audit not possible without patient linkage.\n")

    outdir.joinpath("oauthc_audit.md").write_text("\n".join(lines))


# --------------------------------------------------------------- main ----

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wagner", default="results/analysis/wagner_zeroshot")
    p.add_argument("--slide-table", default="results/data/slide_table.csv")
    p.add_argument("--clinical-full", default="results/data/clinical_table_full.csv")
    p.add_argument("--outdir", default="results/analysis/failure_diagnosis")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_enriched(Path(args.wagner), Path(args.slide_table), Path(args.clinical_full))
    df.to_csv(outdir / "slide_stats.csv", index=False)
    print(f"Enriched slide stats: {len(df)} rows, saved to {outdir/'slide_stats.csv'}")

    # 1a
    ter = tercile_analysis(df, outdir)
    print("\n1a — Wagner AUROC by tissue-area tercile")
    print(ter[ter["stratum"] == "overall"].to_string(index=False))

    # 1c
    coefs = error_profiling(df, outdir)
    print("\n1c — Top 10 covariates (|coef|) for logit(correct ~ X):")
    print(coefs.head(10).to_string(index=False))

    # 1d
    oauthc_audit(df, outdir)
    print(f"\n1d — OAUTHC audit written to {outdir/'oauthc_audit.md'}")

    # Final summary
    summary = {
        "n_slides": int(len(df)),
        "overall_wagner_auroc": _safe_auroc(df["y"], df["p_msih"]),
        "tercile_auroc_overall": (
            ter[ter["stratum"] == "overall"][["tercile", "auroc", "n"]]
            .set_index("tercile")["auroc"].to_dict()
        ),
        "top_error_feature": coefs.iloc[0].to_dict(),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary: {outdir/'summary.json'}")


if __name__ == "__main__":
    main()
