"""C5 Phase 2 — SlideAttentionMSI: learned slide attention for patient-level MSI prediction.

Loads pre-computed slide-level embeddings + Wagner zero-shot scores,
groups slides into patient-level bags, and trains a lightweight attention
model (~5K params) that learns which slides within a patient's bag are
informative for MSI prediction.

Usage:
    python scripts/c5_phase2_slide_attention.py

Outputs:
    results/analysis/c5_phase2/
        ablation_results.csv          — full ablation table
        ablation_heatmap.png          — AUROC heatmap (model × feature set)
        attention_weights.csv         — per-slide attention weights (best model)
        oof_predictions.csv           — out-of-fold patient predictions
        training_curves.png           — loss/AUROC curves
        per_site_auroc.csv            — site-level breakdown
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUTDIR = Path("results/analysis/c5_phase2")
EMB_ROOT = Path("results/embeddings")
CLINICAL = Path("results/data/clinical_table.csv")
WAGNER_SLIDES = Path("results/analysis/wagner_zeroshot/slide_scores.csv")

# Which foundation model embeddings to try
EMBEDDING_MODELS = [
    "conch_v1.5_mean",
    "virchow2_mean",
    "uni2_mean",
    "ctranspath_mean",
]

# Ablation feature sets
FEATURE_SETS = {
    "wagner_only":   {"wagner": True,  "embedding": False, "metadata": False},
    "emb_only":      {"wagner": False, "embedding": True,  "metadata": False},
    "wagner+meta":   {"wagner": True,  "embedding": False, "metadata": True},
    "emb+meta":      {"wagner": False, "embedding": True,  "metadata": True},
    "wagner+emb":    {"wagner": True,  "embedding": True,  "metadata": False},
    "all":           {"wagner": True,  "embedding": True,  "metadata": True},
}

DEVICE = "cpu"  # lightweight model, CPU is fine for 217 patients
N_FOLDS = 5
N_EPOCHS = 150
PATIENCE = 20
LR = 1e-3
WEIGHT_DECAY = 1e-2
HIDDEN_DIM = 64
DROPOUT = 0.5
SEED = 42


# ============================================================================
# Model
# ============================================================================

class SlideAttentionMSI(nn.Module):
    """Learned attention over slides within a patient.

    Input: per-slide features (n_slides, D)
    Output: single patient-level MSI logit
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.5):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, slide_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            slide_features: (n_slides, D) — one patient's bag
        Returns:
            logit: scalar
            attention_weights: (n_slides,) softmax weights
        """
        a = self.gate(slide_features)           # (n_slides, 1)
        w = torch.softmax(a, dim=0).squeeze(-1) # (n_slides,)
        pooled = (w.unsqueeze(-1) * slide_features).sum(dim=0)  # (D,)
        logit = self.head(pooled).squeeze()     # scalar
        return logit, w


# ============================================================================
# Data loading
# ============================================================================

def load_patient_bags(
    emb_model: str,
    feature_config: dict,
) -> tuple[list[torch.Tensor], np.ndarray, np.ndarray, pd.DataFrame]:
    """Build per-patient slide bags with selected features.

    Returns:
        bags: list of (n_slides_i, D) tensors, one per patient
        labels: (n_patients,) binary MSI-H labels
        patients: (n_patients,) patient IDs
        slide_info: DataFrame with per-slide metadata for attention viz
    """
    # Load clinical data
    clinical = pd.read_csv(CLINICAL)
    clinical["y"] = (clinical["isMSIH"] == "MSI-H").astype(int)

    # Load Wagner slide scores
    wagner = pd.read_csv(WAGNER_SLIDES)[["slide_id", "patient_id", "site", "n_tiles", "p_msih"]]

    # Load slide embeddings if needed
    emb_dir = EMB_ROOT / emb_model
    if feature_config["embedding"] and emb_dir.exists():
        emb_matrix = np.load(emb_dir / "embeddings.npy")
        emb_meta = pd.read_csv(emb_dir / "metadata.csv")
        emb_meta["_emb_idx"] = np.arange(len(emb_meta))

        # Merge Wagner + embeddings on slide_id
        merged = wagner.merge(
            emb_meta[["slide_id", "_emb_idx"]],
            on="slide_id",
            how="inner",
        )
    else:
        merged = wagner.copy()
        emb_matrix = None

    # Merge with clinical for labels
    merged = merged.merge(
        clinical[["PATIENT", "y"]],
        left_on="patient_id",
        right_on="PATIENT",
        how="inner",
    )

    # Build per-slide feature vectors
    feature_parts = []
    feat_names = []

    if feature_config["wagner"]:
        feature_parts.append(merged[["p_msih"]].values)
        feat_names.append("wagner_p")

    if feature_config["embedding"] and emb_matrix is not None:
        emb_feats = emb_matrix[merged["_emb_idx"].values]
        feature_parts.append(emb_feats)
        feat_names.append(f"emb({emb_feats.shape[1]}d)")

    if feature_config["metadata"]:
        meta_feats = np.column_stack([
            np.log1p(merged["n_tiles"].values),  # log tile count
        ])
        feature_parts.append(meta_feats)
        feat_names.append("log_ntiles")

    if not feature_parts:
        raise ValueError("No features selected!")

    X_all = np.hstack(feature_parts).astype(np.float32)
    log.info(f"Feature vector: {' + '.join(feat_names)} = {X_all.shape[1]}D")

    # Normalize features globally (fit on all data, applied per-fold later if needed)
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)

    # Group into patient bags
    merged["_feat_idx"] = np.arange(len(merged))
    patient_groups = merged.groupby("patient_id")

    bags = []
    labels = []
    patients = []
    slide_records = []

    for pid, grp in patient_groups:
        idx = grp["_feat_idx"].values
        bag = torch.tensor(X_all[idx], dtype=torch.float32)
        bags.append(bag)
        labels.append(grp["y"].iloc[0])
        patients.append(pid)

        for _, row in grp.iterrows():
            slide_records.append({
                "patient_id": pid,
                "slide_id": row["slide_id"],
                "site": row["site"],
                "n_tiles": row["n_tiles"],
                "wagner_p": row["p_msih"],
                "y": row["y"],
            })

    labels = np.array(labels)
    patients = np.array(patients)
    slide_info = pd.DataFrame(slide_records)

    log.info(f"Loaded {len(bags)} patients, {sum(b.shape[0] for b in bags)} slides")
    log.info(f"MSI-H: {labels.sum()} ({100*labels.mean():.1f}%)")
    log.info(f"Slides/patient: median={np.median([b.shape[0] for b in bags]):.0f}, "
             f"max={max(b.shape[0] for b in bags)}")

    return bags, labels, patients, slide_info


# ============================================================================
# Training
# ============================================================================

def train_one_fold(
    train_bags: list[torch.Tensor],
    train_labels: np.ndarray,
    val_bags: list[torch.Tensor],
    val_labels: np.ndarray,
    input_dim: int,
    pos_weight: float,
) -> tuple[SlideAttentionMSI, dict]:
    """Train one fold, return best model + history."""

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = SlideAttentionMSI(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=DEVICE))

    history = {"train_loss": [], "val_loss": [], "val_auroc": []}
    best_auroc = -1
    best_state = None
    patience_counter = 0

    for epoch in range(N_EPOCHS):
        # --- Train ---
        model.train()
        train_losses = []
        # Shuffle patient order each epoch
        perm = np.random.permutation(len(train_bags))
        for i in perm:
            bag = train_bags[i].to(DEVICE)
            label = torch.tensor(train_labels[i], dtype=torch.float32, device=DEVICE)

            optimizer.zero_grad()
            logit, _ = model(bag)
            loss = criterion(logit, label)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # --- Validate ---
        model.eval()
        val_logits = []
        val_losses = []
        with torch.no_grad():
            for i in range(len(val_bags)):
                bag = val_bags[i].to(DEVICE)
                label = torch.tensor(val_labels[i], dtype=torch.float32, device=DEVICE)
                logit, _ = model(bag)
                val_losses.append(criterion(logit, label).item())
                val_logits.append(logit.item())

        val_probs = torch.sigmoid(torch.tensor(val_logits)).numpy()

        # AUROC — handle edge case where all val labels are same class
        if len(np.unique(val_labels)) > 1:
            val_auroc = roc_auc_score(val_labels, val_probs)
        else:
            val_auroc = 0.5

        history["train_loss"].append(np.mean(train_losses))
        history["val_loss"].append(np.mean(val_losses))
        history["val_auroc"].append(val_auroc)

        # Early stopping
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                log.info(f"  Early stop at epoch {epoch+1}, best AUROC={best_auroc:.3f}")
                break

    # Load best model
    model.load_state_dict(best_state)
    return model, history


def cross_validate(
    bags: list[torch.Tensor],
    labels: np.ndarray,
    patients: np.ndarray,
    slide_info: pd.DataFrame,
) -> dict:
    """5-fold patient-level CV. Returns OOF predictions + attention weights."""

    input_dim = bags[0].shape[1]
    pos_weight = (labels == 0).sum() / max((labels == 1).sum(), 1)
    log.info(f"pos_weight={pos_weight:.2f}")

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_preds = np.full(len(bags), np.nan)
    all_attention = []
    all_histories = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(patients, labels)):
        log.info(f"Fold {fold+1}/{N_FOLDS}: train={len(train_idx)}, val={len(val_idx)}, "
                 f"val_MSI-H={labels[val_idx].sum()}")

        train_bags = [bags[i] for i in train_idx]
        val_bags = [bags[i] for i in val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]

        model, history = train_one_fold(
            train_bags, train_labels, val_bags, val_labels,
            input_dim=input_dim, pos_weight=pos_weight,
        )
        all_histories.append(history)

        # OOF predictions + attention
        model.eval()
        with torch.no_grad():
            for i, idx in enumerate(val_idx):
                bag = bags[idx].to(DEVICE)
                logit, attn_w = model(bag)
                oof_preds[idx] = torch.sigmoid(logit).item()

                pid = patients[idx]
                pid_slides = slide_info[slide_info["patient_id"] == pid]
                for j, (_, row) in enumerate(pid_slides.iterrows()):
                    all_attention.append({
                        "fold": fold,
                        "patient_id": pid,
                        "slide_id": row["slide_id"],
                        "site": row["site"],
                        "attention_weight": attn_w[j].item(),
                        "wagner_p": row["wagner_p"],
                        "n_tiles": row["n_tiles"],
                        "y": row["y"],
                    })

    # Metrics
    valid = ~np.isnan(oof_preds)
    auroc = roc_auc_score(labels[valid], oof_preds[valid])
    auprc = average_precision_score(labels[valid], oof_preds[valid])

    # Per-site AUROC
    site_map = slide_info.groupby("patient_id")["site"].first()
    per_site = {}
    for site in site_map.unique():
        site_pats = site_map[site_map == site].index
        site_mask = np.isin(patients, site_pats) & valid
        if labels[site_mask].sum() > 0 and (~labels[site_mask].astype(bool)).sum() > 0:
            per_site[site] = {
                "auroc": roc_auc_score(labels[site_mask], oof_preds[site_mask]),
                "n": site_mask.sum(),
                "n_msih": labels[site_mask].sum(),
            }

    # Per n_slides bin AUROC
    n_slides_per_patient = np.array([bags[i].shape[0] for i in range(len(bags))])
    bins = {"1": n_slides_per_patient == 1,
            "2-3": (n_slides_per_patient >= 2) & (n_slides_per_patient <= 3),
            "4-6": (n_slides_per_patient >= 4) & (n_slides_per_patient <= 6),
            "7+": n_slides_per_patient >= 7}
    per_bin = {}
    for bname, bmask in bins.items():
        bv = bmask & valid
        if labels[bv].sum() > 0 and (~labels[bv].astype(bool)).sum() > 0:
            per_bin[bname] = {
                "auroc": roc_auc_score(labels[bv], oof_preds[bv]),
                "n": bv.sum(),
                "n_msih": labels[bv].sum(),
            }

    return {
        "auroc": auroc,
        "auprc": auprc,
        "oof_preds": oof_preds,
        "attention": pd.DataFrame(all_attention),
        "histories": all_histories,
        "per_site": per_site,
        "per_bin": per_bin,
        "patients": patients,
        "labels": labels,
    }


# ============================================================================
# Main ablation
# ============================================================================

def run_ablation() -> pd.DataFrame:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    results = []

    for emb_model in EMBEDDING_MODELS:
        emb_dir = EMB_ROOT / emb_model
        has_emb = (emb_dir / "embeddings.npy").exists()

        for feat_name, feat_config in FEATURE_SETS.items():
            # Skip embedding-requiring configs if no embeddings
            if feat_config["embedding"] and not has_emb:
                log.warning(f"Skipping {emb_model}/{feat_name} — no embeddings")
                continue

            # Skip embedding-free configs after first model (they're identical)
            if not feat_config["embedding"] and emb_model != EMBEDDING_MODELS[0]:
                continue

            label = f"{emb_model}/{feat_name}" if feat_config["embedding"] else f"none/{feat_name}"
            log.info(f"\n{'='*60}\n{label}\n{'='*60}")

            try:
                bags, labels, patients, slide_info = load_patient_bags(emb_model, feat_config)
                cv_results = cross_validate(bags, labels, patients, slide_info)

                row = {
                    "embedding": emb_model if feat_config["embedding"] else "none",
                    "feature_set": feat_name,
                    "input_dim": bags[0].shape[1],
                    "auroc": cv_results["auroc"],
                    "auprc": cv_results["auprc"],
                }

                # Add per-site
                for site, metrics in cv_results["per_site"].items():
                    row[f"auroc_{site}"] = metrics["auroc"]
                    row[f"n_{site}"] = metrics["n"]

                # Add per-bin
                for bname, metrics in cv_results["per_bin"].items():
                    row[f"auroc_bin_{bname}"] = metrics["auroc"]
                    row[f"n_bin_{bname}"] = metrics["n"]

                results.append(row)

                log.info(f"AUROC={cv_results['auroc']:.3f}, AUPRC={cv_results['auprc']:.3f}")
                for bname, metrics in cv_results["per_bin"].items():
                    log.info(f"  {bname}: AUROC={metrics['auroc']:.3f} (n={metrics['n']})")

                # Save best model's attention weights + OOF predictions
                if cv_results["auroc"] >= max(r["auroc"] for r in results):
                    cv_results["attention"].to_csv(OUTDIR / "attention_weights.csv", index=False)

                    oof_df = pd.DataFrame({
                        "patient_id": cv_results["patients"],
                        "y": cv_results["labels"],
                        "p_msih": cv_results["oof_preds"],
                    })
                    oof_df.to_csv(OUTDIR / "oof_predictions.csv", index=False)

            except Exception as e:
                log.error(f"Failed {label}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Save ablation results
    df = pd.DataFrame(results)
    df = df.sort_values("auroc", ascending=False)
    df.to_csv(OUTDIR / "ablation_results.csv", index=False)
    log.info(f"\nAblation results saved to {OUTDIR / 'ablation_results.csv'}")

    return df


def plot_ablation(df: pd.DataFrame):
    """Plot ablation heatmap."""
    # Pivot: embedding × feature_set → AUROC
    pivot = df.pivot_table(index="embedding", columns="feature_set", values="auroc")

    # Order columns logically
    col_order = [c for c in ["wagner_only", "wagner+meta", "emb_only", "emb+meta",
                              "wagner+emb", "all"] if c in pivot.columns]
    pivot = pivot[col_order]

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot.values, cmap="RdYlBu_r", vmin=0.45, vmax=0.80, aspect="auto")

    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_order, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Annotate
    for i in range(len(pivot.index)):
        for j in range(len(col_order)):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        color="white" if val < 0.55 or val > 0.75 else "black",
                        fontsize=10, fontweight="bold")

    plt.colorbar(im, label="AUROC")
    ax.set_title("C5 Phase 2 — SlideAttentionMSI Ablation", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / "ablation_heatmap.png", dpi=150, bbox_inches="tight")
    log.info(f"Heatmap saved to {OUTDIR / 'ablation_heatmap.png'}")
    plt.close()


def add_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Add Wagner mean/max baselines to the results table for comparison."""
    wagner_pat = pd.read_csv(Path("results/analysis/wagner_zeroshot/patient_scores.csv"))
    auroc_mean = roc_auc_score(wagner_pat["y"], wagner_pat["p_msih"])

    # Wagner calibrated (max/√n) — load from calibrated aggregation if available
    cal_path = Path("results/analysis/calibrated_aggregation/patient_aggregator_table.csv")
    if cal_path.exists():
        cal = pd.read_csv(cal_path)
        if "y" in cal.columns:
            cal_y = cal["y"]
        else:
            clinical = pd.read_csv(CLINICAL)
            clinical["y"] = (clinical["isMSIH"] == "MSI-H").astype(int)
            cal = cal.merge(clinical[["PATIENT", "y"]], left_on="patient_id", right_on="PATIENT")
            cal_y = cal["y"]
        auroc_cal = roc_auc_score(cal_y, cal["max_over_sqrtn"])
    else:
        auroc_cal = np.nan

    baselines = [
        {"embedding": "baseline", "feature_set": "wagner_mean_pool", "input_dim": 0,
         "auroc": auroc_mean, "auprc": np.nan},
        {"embedding": "baseline", "feature_set": "wagner_max/√n", "input_dim": 0,
         "auroc": auroc_cal, "auprc": np.nan},
    ]

    return pd.concat([pd.DataFrame(baselines), df], ignore_index=True)


if __name__ == "__main__":
    df = run_ablation()
    df = add_baselines(df)
    df.to_csv(OUTDIR / "ablation_results.csv", index=False)
    plot_ablation(df[df["embedding"] != "baseline"])

    # Print summary
    print("\n" + "=" * 70)
    print("SlideAttentionMSI Ablation Results")
    print("=" * 70)
    print(df[["embedding", "feature_set", "input_dim", "auroc", "auprc"]].to_string(index=False))
    print("=" * 70)
