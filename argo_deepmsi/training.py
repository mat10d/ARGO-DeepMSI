"""
Training module for MSI prediction.

Provides simple classifiers and lightweight ViT training on embeddings.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)
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
# Simple classifiers on embeddings
# ============================================================================

def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
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
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        class_weight='balanced',
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    auroc_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
    accuracy_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')

    # Fit final model on all data
    model.fit(X_scaled, y)

    return {
        'model': model,
        'scaler': scaler,
        'auroc_mean': auroc_scores.mean(),
        'auroc_std': auroc_scores.std(),
        'accuracy_mean': accuracy_scores.mean(),
        'accuracy_std': accuracy_scores.std(),
        'cv_auroc_scores': auroc_scores,
        'cv_accuracy_scores': accuracy_scores,
    }


def train_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    n_estimators: int = 100,
    random_state: int = 42,
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
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    auroc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    model.fit(X, y)

    return {
        'model': model,
        'auroc_mean': auroc_scores.mean(),
        'auroc_std': auroc_scores.std(),
        'accuracy_mean': accuracy_scores.mean(),
        'accuracy_std': accuracy_scores.std(),
        'cv_auroc_scores': auroc_scores,
        'cv_accuracy_scores': accuracy_scores,
        'feature_importances': model.feature_importances_,
    }


def train_svm(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
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
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SVC(
        kernel='rbf',
        probability=True,
        random_state=random_state,
        class_weight='balanced',
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    auroc_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
    accuracy_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')

    model.fit(X_scaled, y)

    return {
        'model': model,
        'scaler': scaler,
        'auroc_mean': auroc_scores.mean(),
        'auroc_std': auroc_scores.std(),
        'accuracy_mean': accuracy_scores.mean(),
        'accuracy_std': accuracy_scores.std(),
        'cv_auroc_scores': auroc_scores,
        'cv_accuracy_scores': accuracy_scores,
    }


def compare_classifiers(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
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
    results = []

    logger.info("Training Logistic Regression...")
    lr_results = train_logistic_regression(X, y, n_splits, random_state)
    results.append({
        'classifier': 'Logistic Regression',
        'auroc_mean': lr_results['auroc_mean'],
        'auroc_std': lr_results['auroc_std'],
        'accuracy_mean': lr_results['accuracy_mean'],
        'accuracy_std': lr_results['accuracy_std'],
    })

    logger.info("Training Random Forest...")
    rf_results = train_random_forest(X, y, n_splits, random_state=random_state)
    results.append({
        'classifier': 'Random Forest',
        'auroc_mean': rf_results['auroc_mean'],
        'auroc_std': rf_results['auroc_std'],
        'accuracy_mean': rf_results['accuracy_mean'],
        'accuracy_std': rf_results['accuracy_std'],
    })

    logger.info("Training SVM...")
    svm_results = train_svm(X, y, n_splits, random_state)
    results.append({
        'classifier': 'SVM',
        'auroc_mean': svm_results['auroc_mean'],
        'auroc_std': svm_results['auroc_std'],
        'accuracy_mean': svm_results['accuracy_mean'],
        'accuracy_std': svm_results['accuracy_std'],
    })

    df = pd.DataFrame(results)
    df = df.sort_values('auroc_mean', ascending=False)

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
            hidden_dims: List[int] = [256, 128],
            num_classes: int = 2,
            dropout: float = 0.3,
        ):
            super().__init__()

            layers = []
            prev_dim = input_dim

            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(dropout),
                ])
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
    hidden_dims: List[int] = [256, 128],
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
    history = {'train_loss': [], 'val_loss': [], 'val_auroc': []}

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0

        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size].to(device)
            batch_y = y_train[i:i+batch_size].to(device)

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

        history['train_loss'].append(train_loss / (len(X_train) / batch_size))
        history['val_loss'].append(val_loss)
        history['val_auroc'].append(val_auroc)

        if (epoch + 1) % 20 == 0:
            logger.info(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {history['train_loss'][-1]:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val AUROC: {val_auroc:.4f}")

    return {
        'model': model,
        'history': history,
        'best_val_auroc': max(history['val_auroc']),
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

    save_dict = {'model': model}
    if metadata:
        save_dict['metadata'] = metadata

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
