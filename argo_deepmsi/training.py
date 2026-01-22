"""
Training module for MSI prediction.

Based on recent literature showing simple aggregators beat ABMIL in low-data settings:
- MI-SimpleShot (UNI paper, Nature Medicine 2024)
- SiMLP (Feb 2025, arXiv)
- TITAN (Nature Medicine 2025)

Priority order for ~300 slides:
1. Linear Probe (LogisticRegressionCV with L2)
2. k-NN (zero parameters)
3. MI-SimpleShot (prototype-based)
4. SiMLP (mean pool + 2-layer MLP)
5. ABMIL only if >50 slides per class
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeClassifierCV
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .io_utils import get_models_dir, ensure_dir

logger = logging.getLogger(__name__)


# ============================================================================
# Literature-backed classifiers for low-data settings
# ============================================================================

def train_linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    regularization: Literal["l2", "l1", "elasticnet"] = "l2",
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train linear probe with automatic regularization tuning.

    This is the recommended first approach per UNI/TITAN papers.
    Uses LogisticRegressionCV with L-BFGS solver and 45-point lambda sweep.

    Args:
        X: Slide-level feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
        n_splits: Number of CV folds
        regularization: Type of regularization ("l2", "l1", "elasticnet")
        random_state: Random seed

    Returns:
        Dictionary with model, scaler, and metrics
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # LogisticRegressionCV with extensive lambda sweep (as in foundation model papers)
    Cs = np.logspace(-6, 5, 45)  # 45 values from 10^-6 to 10^5

    if regularization == "l2":
        model = LogisticRegressionCV(
            Cs=Cs,
            cv=n_splits,
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1,
        )
    elif regularization == "l1":
        model = LogisticRegressionCV(
            Cs=Cs,
            cv=n_splits,
            penalty='l1',
            solver='saga',
            max_iter=1000,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1,
        )
    else:  # elasticnet
        model = LogisticRegressionCV(
            Cs=Cs,
            cv=n_splits,
            penalty='elasticnet',
            solver='saga',
            l1_ratios=[0.1, 0.5, 0.9],
            max_iter=1000,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1,
        )

    model.fit(X_scaled, y)

    # Get CV predictions for metrics
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_proba = cross_val_predict(model, X_scaled, y, cv=cv, method='predict_proba')[:, 1]
    y_pred = cross_val_predict(model, X_scaled, y, cv=cv)

    auroc = roc_auc_score(y, y_proba)
    auprc = average_precision_score(y, y_proba)
    accuracy = accuracy_score(y, y_pred)

    logger.info(f"Linear Probe ({regularization}): AUROC={auroc:.3f}, AUPRC={auprc:.3f}, Acc={accuracy:.3f}")
    logger.info(f"  Best C (inverse regularization): {model.C_[0]:.4f}")

    return {
        'model': model,
        'scaler': scaler,
        'auroc': auroc,
        'auprc': auprc,
        'accuracy': accuracy,
        'best_C': model.C_[0],
        'regularization': regularization,
    }


def train_knn(
    X: np.ndarray,
    y: np.ndarray,
    n_neighbors: int = 20,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train k-Nearest Neighbors classifier.

    Zero trainable parameters - competitive baseline per foundation model papers.
    Default k=20 as commonly used in the literature.

    Args:
        X: Feature matrix
        y: Labels
        n_neighbors: Number of neighbors (default 20 per literature)
        n_splits: Number of CV folds
        random_state: Random seed

    Returns:
        Dictionary with model and metrics
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        metric='cosine',  # Cosine similarity for embeddings
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_proba = cross_val_predict(model, X_scaled, y, cv=cv, method='predict_proba')[:, 1]
    y_pred = cross_val_predict(model, X_scaled, y, cv=cv)

    model.fit(X_scaled, y)

    auroc = roc_auc_score(y, y_proba)
    auprc = average_precision_score(y, y_proba)
    accuracy = accuracy_score(y, y_pred)

    logger.info(f"k-NN (k={n_neighbors}): AUROC={auroc:.3f}, AUPRC={auprc:.3f}, Acc={accuracy:.3f}")

    return {
        'model': model,
        'scaler': scaler,
        'auroc': auroc,
        'auprc': auprc,
        'accuracy': accuracy,
        'n_neighbors': n_neighbors,
    }


def train_mi_simpleshot(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """MI-SimpleShot: Prototype-based nearest neighbor classifier.

    From UNI paper (Nature Medicine 2024). Creates class prototypes from
    training examples and uses nearest-neighbor matching. Works well with
    1-4 slides per class.

    Args:
        X: Feature matrix (already slide-level)
        y: Labels
        n_splits: Number of CV folds
        random_state: Random seed

    Returns:
        Dictionary with prototypes and metrics
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    all_proba = np.zeros(len(y))
    all_pred = np.zeros(len(y), dtype=int)

    for train_idx, test_idx in cv.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Create class prototypes (mean of each class)
        classes = np.unique(y_train)
        prototypes = {}
        for c in classes:
            prototypes[c] = X_train[y_train == c].mean(axis=0)

        # Predict using cosine similarity to prototypes
        for i, x in enumerate(X_test):
            similarities = {}
            for c, proto in prototypes.items():
                # Cosine similarity
                sim = np.dot(x, proto) / (np.linalg.norm(x) * np.linalg.norm(proto) + 1e-8)
                similarities[c] = sim

            # Softmax over similarities for probabilities
            sims = np.array([similarities[c] for c in sorted(classes)])
            probs = np.exp(sims) / np.exp(sims).sum()

            all_proba[test_idx[i]] = probs[1] if len(classes) == 2 else probs.max()
            all_pred[test_idx[i]] = sorted(classes)[np.argmax(sims)]

    auroc = roc_auc_score(y, all_proba)
    auprc = average_precision_score(y, all_proba)
    accuracy = accuracy_score(y, all_pred)

    # Final prototypes on all data
    final_prototypes = {}
    for c in np.unique(y):
        final_prototypes[c] = X_scaled[y == c].mean(axis=0)

    logger.info(f"MI-SimpleShot: AUROC={auroc:.3f}, AUPRC={auprc:.3f}, Acc={accuracy:.3f}")

    return {
        'prototypes': final_prototypes,
        'scaler': scaler,
        'auroc': auroc,
        'auprc': auprc,
        'accuracy': accuracy,
    }


# ============================================================================
# SiMLP: Simple MLP (Feb 2025 arXiv)
# ============================================================================

if TORCH_AVAILABLE:
    class SiMLP(nn.Module):
        """Simple 2-layer MLP from SiMLP paper (Feb 2025).

        Mean pooling + 2-layer MLP with ReLU.
        Consistently beats ABMIL in few-shot settings (K=1,5,10,20,50).
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 256,
            num_classes: int = 2,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, num_classes)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            # x: (batch, input_dim) - already mean-pooled slide embeddings
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x


    class AttentionMIL(nn.Module):
        """Attention-based MIL for bag-level classification.

        WARNING: Only use if you have >50 slides per class.
        Overfits in low-data settings.
        """

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
            h = self.feature_extractor(x)
            a = self.attention(h)
            a = torch.softmax(a, dim=1)
            z = (a * h).sum(dim=1)
            return self.classifier(z)


def train_simlp(
    X: np.ndarray,
    y: np.ndarray,
    hidden_dim: int = 256,
    n_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    n_splits: int = 5,
    device: str = "cuda",
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train SiMLP (Simple MLP) on slide embeddings.

    From SiMLP paper (Feb 2025). Mean pooling + 2-layer MLP.
    Better than ABMIL for few-shot learning.

    Args:
        X: Slide-level embeddings (n_samples, n_features)
        y: Labels
        hidden_dim: Hidden layer dimension
        n_epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: L2 regularization
        n_splits: CV folds for evaluation
        device: cuda or cpu
        random_state: Random seed

    Returns:
        Dictionary with model and metrics
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for SiMLP")

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_proba = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y)):
        X_train = torch.FloatTensor(X_scaled[train_idx]).to(device)
        y_train = torch.LongTensor(y[train_idx]).to(device)
        X_val = torch.FloatTensor(X_scaled[val_idx]).to(device)

        model = SiMLP(
            input_dim=X.shape[1],
            hidden_dim=hidden_dim,
            num_classes=len(np.unique(y)),
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        # Training
        model.train()
        for epoch in range(n_epochs):
            perm = torch.randperm(len(X_train))
            for i in range(0, len(X_train), batch_size):
                idx = perm[i:i+batch_size]
                optimizer.zero_grad()
                loss = criterion(model(X_train[idx]), y_train[idx])
                loss.backward()
                optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            proba = torch.softmax(model(X_val), dim=1)[:, 1].cpu().numpy()
            all_proba[val_idx] = proba

    auroc = roc_auc_score(y, all_proba)
    auprc = average_precision_score(y, all_proba)
    accuracy = accuracy_score(y, (all_proba > 0.5).astype(int))

    # Train final model on all data
    X_all = torch.FloatTensor(X_scaled).to(device)
    y_all = torch.LongTensor(y).to(device)

    final_model = SiMLP(
        input_dim=X.shape[1],
        hidden_dim=hidden_dim,
        num_classes=len(np.unique(y)),
    ).to(device)

    optimizer = torch.optim.AdamW(final_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    final_model.train()
    for epoch in range(n_epochs):
        perm = torch.randperm(len(X_all))
        for i in range(0, len(X_all), batch_size):
            idx = perm[i:i+batch_size]
            optimizer.zero_grad()
            loss = criterion(final_model(X_all[idx]), y_all[idx])
            loss.backward()
            optimizer.step()

    logger.info(f"SiMLP: AUROC={auroc:.3f}, AUPRC={auprc:.3f}, Acc={accuracy:.3f}")

    return {
        'model': final_model,
        'scaler': scaler,
        'auroc': auroc,
        'auprc': auprc,
        'accuracy': accuracy,
    }


# ============================================================================
# Compare all methods
# ============================================================================

def compare_low_data_methods(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    include_simlp: bool = True,
) -> pd.DataFrame:
    """Compare all recommended methods for low-data settings.

    Runs in order of complexity:
    1. Linear Probe (L2)
    2. Linear Probe (ElasticNet)
    3. k-NN (k=20)
    4. MI-SimpleShot
    5. SiMLP (if include_simlp=True)

    Args:
        X: Slide-level embeddings
        y: Labels
        n_splits: CV folds
        random_state: Random seed
        include_simlp: Whether to include SiMLP (requires PyTorch)

    Returns:
        DataFrame with results sorted by AUROC
    """
    results = []

    # 1. Linear Probe (L2) - RECOMMENDED FIRST
    logger.info("\n1. Training Linear Probe (L2)...")
    lp_l2 = train_linear_probe(X, y, n_splits, "l2", random_state)
    results.append({
        'method': 'Linear Probe (L2)',
        'auroc': lp_l2['auroc'],
        'auprc': lp_l2['auprc'],
        'accuracy': lp_l2['accuracy'],
        'params': f"C={lp_l2['best_C']:.4f}",
    })

    # 2. Linear Probe (ElasticNet)
    logger.info("\n2. Training Linear Probe (ElasticNet)...")
    lp_en = train_linear_probe(X, y, n_splits, "elasticnet", random_state)
    results.append({
        'method': 'Linear Probe (ElasticNet)',
        'auroc': lp_en['auroc'],
        'auprc': lp_en['auprc'],
        'accuracy': lp_en['accuracy'],
        'params': f"C={lp_en['best_C']:.4f}",
    })

    # 3. k-NN
    logger.info("\n3. Training k-NN (k=20)...")
    knn = train_knn(X, y, n_neighbors=20, n_splits=n_splits, random_state=random_state)
    results.append({
        'method': 'k-NN (k=20)',
        'auroc': knn['auroc'],
        'auprc': knn['auprc'],
        'accuracy': knn['accuracy'],
        'params': 'k=20, cosine',
    })

    # 4. MI-SimpleShot
    logger.info("\n4. Training MI-SimpleShot...")
    simpleshot = train_mi_simpleshot(X, y, n_splits, random_state)
    results.append({
        'method': 'MI-SimpleShot',
        'auroc': simpleshot['auroc'],
        'auprc': simpleshot['auprc'],
        'accuracy': simpleshot['accuracy'],
        'params': 'prototype-based',
    })

    # 5. SiMLP (optional)
    if include_simlp and TORCH_AVAILABLE:
        logger.info("\n5. Training SiMLP...")
        simlp = train_simlp(X, y, n_splits=n_splits, random_state=random_state)
        results.append({
            'method': 'SiMLP',
            'auroc': simlp['auroc'],
            'auprc': simlp['auprc'],
            'accuracy': simlp['accuracy'],
            'params': 'hidden=256',
        })

    df = pd.DataFrame(results)
    df = df.sort_values('auroc', ascending=False)

    logger.info("\n" + "="*70)
    logger.info("RESULTS (sorted by AUROC)")
    logger.info("="*70)
    logger.info(df.to_string(index=False))

    return df


# ============================================================================
# Legacy functions (kept for backwards compatibility)
# ============================================================================

def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Legacy function - use train_linear_probe instead."""
    result = train_linear_probe(X, y, n_splits, "l2", random_state)
    return {
        'model': result['model'],
        'scaler': result['scaler'],
        'auroc_mean': result['auroc'],
        'auroc_std': 0.0,
        'accuracy_mean': result['accuracy'],
        'accuracy_std': 0.0,
    }


def train_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    n_estimators: int = 100,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train random forest (not recommended for low-data, kept for comparison)."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]

    model.fit(X, y)

    return {
        'model': model,
        'auroc_mean': roc_auc_score(y, y_proba),
        'auroc_std': 0.0,
        'accuracy_mean': accuracy_score(y, (y_proba > 0.5).astype(int)),
        'accuracy_std': 0.0,
        'feature_importances': model.feature_importances_,
    }


def train_svm(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train SVM (kept for comparison)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SVC(kernel='rbf', probability=True, random_state=random_state, class_weight='balanced')

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_proba = cross_val_predict(model, X_scaled, y, cv=cv, method='predict_proba')[:, 1]

    model.fit(X_scaled, y)

    return {
        'model': model,
        'scaler': scaler,
        'auroc_mean': roc_auc_score(y, y_proba),
        'auroc_std': 0.0,
        'accuracy_mean': accuracy_score(y, (y_proba > 0.5).astype(int)),
        'accuracy_std': 0.0,
    }


def compare_classifiers(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Legacy function - use compare_low_data_methods instead."""
    return compare_low_data_methods(X, y, n_splits, random_state, include_simlp=TORCH_AVAILABLE)


# ============================================================================
# Model persistence
# ============================================================================

def save_model(
    model: Any,
    model_name: str,
    metadata: Optional[Dict] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """Save trained model to disk."""
    import joblib

    if output_dir is None:
        output_dir = get_models_dir()
    ensure_dir(output_dir)

    model_path = output_dir / f"{model_name}.joblib"

    save_dict = {'model': model}
    if metadata:
        save_dict['metadata'] = metadata

    joblib.dump(save_dict, model_path)
    logger.info(f"Saved model: {model_path}")

    return model_path


def load_model(model_path: Path) -> Dict[str, Any]:
    """Load trained model from disk."""
    import joblib
    return joblib.load(model_path)
