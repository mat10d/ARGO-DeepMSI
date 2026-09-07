"""
Training module for MSI prediction.

Provides simple classifiers and lightweight ViT training on embeddings.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .io_utils import get_models_dir, ensure_dir

logger = logging.getLogger(__name__)


# ============================================================================
# Data loading with robust patient-slide matching
# ============================================================================


def load_training_data(
    embeddings_dir: Path,
    clinical_table: Path,
    label_column: str = "isMSIH",
    positive_label: str = "MSI-H",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load embeddings and labels with robust patient-slide matching.

    Args:
        embeddings_dir: Directory with embeddings.npy and metadata.csv
        clinical_table: clinical_table.csv with PATIENT and labels
        label_column: Column name for MSI status
        positive_label: Value indicating MSI-H

    Returns:
        X: Embedding matrix (n_samples, n_features)
        y: Binary labels (n_samples,)
        merged: DataFrame with matched records
    """
    # Load data — prefer AnnData (scverse-native) when present, fall back to npy+csv
    h5ad_path = embeddings_dir / "embeddings.h5ad"
    if h5ad_path.exists():
        import anndata as ad

        adata = ad.read_h5ad(h5ad_path)
        embeddings = np.asarray(adata.X)
        metadata = adata.obs.reset_index(drop=True)
        logger.info(f"Loaded AnnData embeddings: {h5ad_path}")
    else:
        embeddings = np.load(embeddings_dir / "embeddings.npy")
        metadata = pd.read_csv(embeddings_dir / "metadata.csv").reset_index(drop=True)
    clinical = pd.read_csv(clinical_table)

    logger.info(f"Loaded {len(embeddings)} embeddings from {embeddings_dir}")
    logger.info(f"Loaded {len(clinical)} clinical records")

    if len(metadata) != len(embeddings):
        raise ValueError(
            f"Row count mismatch: metadata has {len(metadata)} rows but "
            f"embeddings array has {len(embeddings)} rows."
        )

    # Preserve the source row explicitly. pandas.merge creates a new RangeIndex,
    # which does not identify the embedding row when metadata is unmatched or
    # the merge changes row order.
    metadata = metadata.copy()
    metadata["_embedding_row"] = np.arange(len(metadata))
    merged = metadata.merge(
        clinical,
        left_on="patient_id",
        right_on="PATIENT",
        how="inner",
        validate="many_to_one",  # Each patient can have multiple slides
    )

    if len(merged) == 0:
        raise ValueError(
            "No matching records between metadata and clinical table! "
            "Check that:\n"
            "  - metadata.csv has 'patient_id' column\n"
            "  - clinical_table.csv has 'PATIENT' column\n"
            "  - Values match (case-sensitive)"
        )

    embedding_rows = merged.pop("_embedding_row").to_numpy(dtype=int)
    X = embeddings[embedding_rows]

    # Extract labels
    y = (merged[label_column] == positive_label).astype(int).values

    # Log statistics
    logger.info(f"Matched {len(X)} samples")
    logger.info("Label distribution:")
    logger.info(f"  {positive_label}: {y.sum()} ({100 * y.sum() / len(y):.1f}%)")
    logger.info(f"  Other: {len(y) - y.sum()} ({100 * (len(y) - y.sum()) / len(y):.1f}%)")

    # Warn if severely imbalanced
    if y.sum() < 5 or (len(y) - y.sum()) < 5:
        logger.warning(
            "Severely imbalanced dataset! Consider stratified sampling or collecting more data."
        )

    return X, y, merged


# ============================================================================
# Simple classifiers on embeddings
# ============================================================================


def _cross_validate_and_fit(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    groups: Optional[np.ndarray],
    n_splits: int,
    random_state: int,
    scale_features: bool,
) -> Dict[str, Any]:
    """Evaluate with grouped CV, then fit the classifier on all observations.

    Preprocessing lives in the estimator pipeline so it is fitted independently
    inside each fold. Both metrics are collected in one CV pass.
    """
    if groups is None:
        raise ValueError(
            "`groups` (patient IDs) must be provided to avoid patient-level data leakage."
        )

    estimator = (
        Pipeline([("scaler", StandardScaler()), ("classifier", model)]) if scale_features else model
    )
    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    scores = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        groups=groups,
        scoring={"auroc": "roc_auc", "accuracy": "accuracy"},
    )
    estimator.fit(X, y)

    auroc_scores = scores["test_auroc"]
    accuracy_scores = scores["test_accuracy"]
    result = {
        "model": estimator,
        "auroc_mean": auroc_scores.mean(),
        "auroc_std": auroc_scores.std(),
        "accuracy_mean": accuracy_scores.mean(),
        "accuracy_std": accuracy_scores.std(),
        "cv_auroc_scores": auroc_scores,
        "cv_accuracy_scores": accuracy_scores,
    }
    if scale_features:
        # Retain the original model/scaler fields for callers that use them
        # separately, and expose the safer raw-input pipeline as well.
        result["model"] = estimator.named_steps["classifier"]
        result["scaler"] = estimator.named_steps["scaler"]
        result["pipeline"] = estimator
    return result


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    n_splits: int = 5,
    random_state: int = 42,
    C: float = 1.0,
    solver: str = "lbfgs",
    max_iter: int = 1000,
    class_weight: str | dict | None = "balanced",
) -> Dict[str, Any]:
    """Train logistic regression with cross-validation.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
        n_splits: Number of CV folds
        random_state: Random seed

    Returns:
        Dictionary with model and metrics
    """
    model = LogisticRegression(
        C=C,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
        class_weight=class_weight,
    )

    return _cross_validate_and_fit(
        model,
        X,
        y,
        groups=groups,
        n_splits=n_splits,
        random_state=random_state,
        scale_features=True,
    )


def train_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    n_splits: int = 5,
    n_estimators: int = 100,
    random_state: int = 42,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    max_features: str | float | int | None = "sqrt",
    class_weight: str | dict | None = "balanced",
) -> Dict[str, Any]:
    """Train random forest with cross-validation.

    Args:
        X: Feature matrix
        y: Labels
        n_splits: Number of CV folds
        n_estimators: Number of trees
        random_state: Random seed

    Returns:
        Dictionary with model and metrics
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1,
    )

    result = _cross_validate_and_fit(
        model,
        X,
        y,
        groups=groups,
        n_splits=n_splits,
        random_state=random_state,
        scale_features=False,
    )
    result["feature_importances"] = result["model"].feature_importances_
    return result


def train_svm(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    n_splits: int = 5,
    random_state: int = 42,
    C: float = 1.0,
    kernel: str = "rbf",
    gamma: str | float = "scale",
    class_weight: str | dict | None = "balanced",
) -> Dict[str, Any]:
    """Train SVM with cross-validation.

    Args:
        X: Feature matrix
        y: Labels
        n_splits: Number of CV folds
        random_state: Random seed

    Returns:
        Dictionary with model and metrics
    """
    model = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        probability=True,
        random_state=random_state,
        class_weight=class_weight,
    )

    return _cross_validate_and_fit(
        model,
        X,
        y,
        groups=groups,
        n_splits=n_splits,
        random_state=random_state,
        scale_features=True,
    )


def compare_classifiers(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    n_splits: int = 5,
    random_state: int = 42,
    classifiers: Sequence[str] = ("logistic", "random_forest", "svm"),
    classifier_params: Mapping[str, Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """Compare multiple classifiers on the same data.

    Args:
        X: Feature matrix
        y: Labels
        n_splits: Number of CV folds
        random_state: Random seed

    Returns:
        DataFrame comparing classifier performance
    """
    if groups is None:
        raise ValueError(
            "`groups` (patient IDs) must be provided to avoid patient-level data leakage."
        )

    trainers = {
        "logistic": ("Logistic Regression", train_logistic_regression),
        "random_forest": ("Random Forest", train_random_forest),
        "svm": ("SVM", train_svm),
    }
    if not classifiers:
        raise ValueError("Select at least one classifier")
    unknown = set(classifiers) - trainers.keys()
    if unknown:
        raise ValueError(f"Unknown classifiers: {sorted(unknown)}. Available: {sorted(trainers)}")
    classifier_params = classifier_params or {}
    unused_params = set(classifier_params) - set(classifiers)
    if unused_params:
        raise ValueError(f"Parameters supplied for unselected classifiers: {sorted(unused_params)}")
    metric_names = ("auroc_mean", "auroc_std", "accuracy_mean", "accuracy_std")
    results = []
    for classifier_key in classifiers:
        classifier_name, trainer = trainers[classifier_key]
        logger.info("Training %s...", classifier_name)
        metrics = trainer(
            X,
            y,
            groups=groups,
            n_splits=n_splits,
            random_state=random_state,
            **classifier_params.get(classifier_key, {}),
        )
        results.append(
            {"classifier": classifier_name, **{name: metrics[name] for name in metric_names}}
        )

    df = pd.DataFrame(results).sort_values("auroc_mean", ascending=False)

    logger.info("\nClassifier Comparison:")
    logger.info(df.to_string(index=False))

    return df


# ============================================================================
# Lightweight MLP classifier
# ============================================================================

if TORCH_AVAILABLE:

    class MLPClassifier(nn.Module):
        """Simple MLP for embedding classification."""

        def __init__(
            self,
            input_dim: int,
            hidden_dims: Sequence[int] = (256, 128),
            num_classes: int = 2,
            dropout: float = 0.3,
        ):
            super().__init__()

            layers = []
            prev_dim = input_dim

            for hidden_dim in hidden_dims:
                layers.extend(
                    [
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(),
                        nn.BatchNorm1d(hidden_dim),
                        nn.Dropout(dropout),
                    ]
                )
                prev_dim = hidden_dim

            layers.append(nn.Linear(prev_dim, num_classes))

            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    class AttentionMIL(nn.Module):
        """Attention-based MIL for bag-level classification."""

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 256,
            attention_dim: int = 128,
            num_classes: int = 2,
            dropout: float = 0.3,
        ):
            super().__init__()

            self.feature_extractor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, attention_dim),
                nn.Tanh(),
                nn.Linear(attention_dim, 1),
            )

            self.classifier = nn.Linear(hidden_dim, num_classes)

        def forward(self, x):
            # x: (batch, n_instances, input_dim)
            h = self.feature_extractor(x)  # (batch, n_instances, hidden_dim)

            # Attention weights
            a = self.attention(h)  # (batch, n_instances, 1)
            a = torch.softmax(a, dim=1)

            # Weighted sum
            z = (a * h).sum(dim=1)  # (batch, hidden_dim)

            return self.classifier(z)


def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    hidden_dims: Sequence[int] = (256, 128),
    n_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str = "cuda",
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train MLP classifier on embeddings.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels
        hidden_dims: Hidden layer dimensions
        n_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device for training
        random_state: Random seed

    Returns:
        Dictionary with model and training history
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed")

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    # Split train/val
    n_val = int(0.2 * len(X))
    indices = torch.randperm(len(X))
    train_idx, val_idx = indices[n_val:], indices[:n_val]

    X_train, y_train = X_tensor[train_idx], y_tensor[train_idx]
    X_val, y_val = X_tensor[val_idx], y_tensor[val_idx]

    # Model
    model = MLPClassifier(
        input_dim=X.shape[1],
        hidden_dims=hidden_dims,
        num_classes=len(np.unique(y)),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    history = {"train_loss": [], "val_loss": [], "val_auroc": []}

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0

        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i : i + batch_size].to(device)
            batch_y = y_train[i : i + batch_size].to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val.to(device))
            val_loss = criterion(val_outputs, y_val.to(device)).item()
            val_probs = torch.softmax(val_outputs, dim=1)[:, 1].cpu().numpy()
            val_auroc = roc_auc_score(y_val.numpy(), val_probs)

        history["train_loss"].append(train_loss / (len(X_train) / batch_size))
        history["val_loss"].append(val_loss)
        history["val_auroc"].append(val_auroc)

        if (epoch + 1) % 20 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{n_epochs} - Train Loss: {history['train_loss'][-1]:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val AUROC: {val_auroc:.4f}"
            )

    return {
        "model": model,
        "history": history,
        "best_val_auroc": max(history["val_auroc"]),
    }


# ============================================================================
# Model persistence
# ============================================================================


def save_model(
    model: Any,
    model_name: str,
    metadata: Optional[Dict] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """Save trained model to disk.

    Args:
        model: Trained model (sklearn or PyTorch)
        model_name: Name for the saved model
        metadata: Optional metadata to save alongside
        output_dir: Output directory

    Returns:
        Path to saved model
    """
    import joblib

    if output_dir is None:
        output_dir = get_models_dir()
    ensure_dir(output_dir)

    model_path = output_dir / f"{model_name}.joblib"

    save_dict = {"model": model}
    if metadata:
        save_dict["metadata"] = metadata

    joblib.dump(save_dict, model_path)
    logger.info(f"Saved model: {model_path}")

    return model_path


def load_model(model_path: Path) -> Dict[str, Any]:
    """Load trained model from disk.

    Args:
        model_path: Path to saved model

    Returns:
        Dictionary with model and optional metadata
    """
    import joblib

    return joblib.load(model_path)
