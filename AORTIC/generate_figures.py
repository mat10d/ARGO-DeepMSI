#!/usr/bin/env python3
"""
Generate publication-quality figures for AORTIC MSI prediction results.

This script processes cross-validation results from CTransPath and H-Optimus-0 models,
as well as results from the pre-trained HistoBistro model, and generates:
1. ROC curves with AUROC values
2. NPV comparison charts
3. NPV vs threshold curves
4. Clinical impact visualizations

Author: Generated for AORTIC project
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'


def load_stamp_predictions(model_name, n_folds=3):
    """
    Load predictions from STAMP cross-validation folds.

    Args:
        model_name: Name of model directory (e.g., 'ctranspath', 'h-optimus-0')
        n_folds: Number of CV folds

    Returns:
        List of dictionaries containing y_true, y_pred, y_proba for each fold
    """
    folds = []
    for i in range(n_folds):
        try:
            df = pd.read_csv(f'{model_name}/patient-preds_{i}.csv')
            # Convert MSI-H/MSS to binary (1 = MSI-H, 0 = MSS)
            y_true = (df['isMSIH'] == 'MSI-H').astype(int)
            y_proba = df['isMSIH_MSI-H'].values
            y_pred = (y_proba >= 0.5).astype(int)

            folds.append({
                'y_true': y_true,
                'y_pred': y_pred,
                'y_proba': y_proba,
                'patients': df['PATIENT'].values
            })
        except FileNotFoundError:
            print(f"Warning: Could not find {model_name}/patient-preds_{i}.csv")

    return folds


def load_histobistro_predictions():
    """
    Load predictions from HistoBistro model.

    Note: HistoBistro stores logits as strings like 'tensor(0.6085)'.
    We extract these, convert to probabilities using sigmoid.

    Returns:
        Dictionary containing y_true, y_pred, y_proba
    """
    df = pd.read_csv('histobistro/histobistro_outputs_all.csv')

    # HistoBistro uses 1 = MSI-H, 0 = MSS
    y_true = df['ground_truth'].astype(int)

    # Extract logits from string format 'tensor(X.XXXX)'
    def extract_logit(logit_str):
        """Extract numeric value from 'tensor(X.XXXX)' string"""
        try:
            value = str(logit_str).replace('tensor(', '').replace(')', '')
            return float(value)
        except:
            return np.nan

    logits = df['logits'].apply(extract_logit).values

    # Convert logits to probabilities using sigmoid function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    y_proba = sigmoid(logits)
    y_pred = (y_proba >= 0.5).astype(int)

    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'patients': df['patient'].values
    }


def load_cnn_predictions():
    """
    Load predictions from CNN models (TCGA and DACHS are two different pre-trained models).

    Note:
    - TCGA and DACHS are two different CNN models tested on the same 94 patients
    - CNN uses inverted labels (0 = MSI-H, 1 = MSS), so we convert to standard
    - We treat them like separate runs (similar to CV folds)

    Returns:
        List of dictionaries for each CNN model (similar to CV folds structure)
    """
    models = []

    for model_name, file_name in [('TCGA', 'CNN/TEST_RESULT_PATIENT_BASED_FULL_tcga.csv'),
                                    ('DACHS', 'CNN/TEST_RESULT_PATIENT_BASED_FULL_dachs.csv')]:
        df = pd.read_csv(file_name)

        # Convert from CNN format (0 = MSI-H, 1 = MSS) to standard format (1 = MSI-H, 0 = MSS)
        y_true = (1 - df['yTrue']).astype(int)
        y_proba = df['MSIH'].values  # Probability of MSI-H
        y_pred = (y_proba >= 0.5).astype(int)

        models.append({
            'y_true': y_true,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'patients': df['PATIENT'].values,
            'model_name': model_name
        })

    return models


def aggregate_cv_folds(folds):
    """
    Aggregate predictions from all CV folds into a single set.
    Each patient appears exactly once (in their test fold).

    Args:
        folds: List of fold dictionaries

    Returns:
        Dictionary with aggregated predictions
    """
    all_y_true = []
    all_y_proba = []
    all_y_pred = []
    all_patients = []

    for fold in folds:
        all_y_true.extend(fold['y_true'])
        all_y_proba.extend(fold['y_proba'])
        all_y_pred.extend(fold['y_pred'])
        all_patients.extend(fold['patients'])

    return {
        'y_true': np.array(all_y_true),
        'y_pred': np.array(all_y_pred),
        'y_proba': np.array(all_y_proba),
        'patients': np.array(all_patients)
    }


def calculate_metrics(y_true, y_proba, threshold=0.5):
    """
    Calculate comprehensive performance metrics.

    NPV (Negative Predictive Value) Calculation:
    ---------------------------------------------
    NPV = TN / (TN + FN)

    Where:
    - TN (True Negatives) = Patients predicted MSS who are truly MSS ✓
    - FN (False Negatives) = Patients predicted MSS who are actually MSI-H ✗ (DANGEROUS!)
    - TP (True Positives) = Patients predicted MSI-H who are truly MSI-H ✓
    - FP (False Positives) = Patients predicted MSI-H who are actually MSS ✗

    NPV answers the question: "If I predict a patient is MSS (negative), what is the
    probability they are truly MSS?"

    Example: NPV = 0.865 means:
    - Out of 100 patients predicted MSS, 86.5 are truly MSS
    - BUT 13.5 are actually MSI-H (missed cases!)

    For a rule-out test, we want NPV > 0.99 (less than 1% missed MSI-H cases)

    Args:
        y_true: True labels (1 = MSI-H, 0 = MSS)
        y_proba: Predicted probabilities for MSI-H
        threshold: Decision threshold (default 0.5)

    Returns:
        Dictionary of metrics
    """
    y_pred = (y_proba >= threshold).astype(int)

    # Confusion matrix
    # Format: [[TN, FP],
    #          [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Core metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall, TPR
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision, Positive Predictive Value
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value

    # AUC
    try:
        auroc = roc_auc_score(y_true, y_proba)
    except:
        auroc = np.nan

    return {
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'PPV': ppv,
        'NPV': npv,
        'AUROC': auroc,
        'Threshold': threshold
    }


def calculate_npv_vs_threshold(y_true, y_proba, thresholds=None):
    """
    Calculate NPV across different thresholds.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        thresholds: Array of thresholds to test

    Returns:
        Arrays of thresholds, NPVs, sensitivities, specificities
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)

    npvs = []
    sensitivities = []
    specificities = []

    for thresh in thresholds:
        metrics = calculate_metrics(y_true, y_proba, threshold=thresh)
        npvs.append(metrics['NPV'])
        sensitivities.append(metrics['Sensitivity'])
        specificities.append(metrics['Specificity'])

    return thresholds, np.array(npvs), np.array(sensitivities), np.array(specificities)


def plot_roc_curves(models_data, output_path='roc_curves.png'):
    """
    Plot ROC curves for all models.

    Args:
        models_data: Dictionary mapping model names to their data
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = {'HistoBistro': '#E74C3C', 'CTransPath': '#3498DB', 'H-Optimus-0': '#2ECC71', 'CNN': '#9B59B6'}

    for model_name, data in models_data.items():
        if 'folds' in data:
            # Plot individual folds with transparency
            for i, fold in enumerate(data['folds']):
                fpr, tpr, _ = roc_curve(fold['y_true'], fold['y_proba'])
                fold_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=colors.get(model_name, 'gray'),
                       alpha=0.2, linewidth=1)

            # Plot aggregated curve
            agg = data['aggregated']
            fpr, tpr, _ = roc_curve(agg['y_true'], agg['y_proba'])
            roc_auc = auc(fpr, tpr)

            # Calculate 95% CI for AUROC across folds
            fold_aucs = [roc_auc_score(f['y_true'], f['y_proba']) for f in data['folds']]
            ci_lower, ci_upper = np.percentile(fold_aucs, [2.5, 97.5])

            label = f'{model_name} (AUC={roc_auc:.3f} [{ci_lower:.3f}-{ci_upper:.3f}])'
            ax.plot(fpr, tpr, color=colors.get(model_name, 'gray'),
                   linewidth=2.5, label=label)
        else:
            # Single model (HistoBistro)
            fpr, tpr, _ = roc_curve(data['y_true'], data['y_proba'])
            roc_auc = auc(fpr, tpr)
            label = f'{model_name} (AUC={roc_auc:.3f})'
            ax.plot(fpr, tpr, color=colors.get(model_name, 'gray'),
                   linewidth=2.5, label=label)

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')

    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curves - MSI Status Prediction', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved ROC curve to {output_path}")
    plt.close()


def plot_npv_comparison(models_data, threshold=0.5, output_path='npv_comparison.png'):
    """
    Plot NPV comparison across models with error bars.

    Args:
        models_data: Dictionary mapping model names to their data
        threshold: Decision threshold for NPV calculation
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    model_names = []
    npv_means = []
    npv_errors = []
    colors_list = []

    colors = {'HistoBistro': '#E74C3C', 'CTransPath': '#3498DB', 'H-Optimus-0': '#2ECC71', 'CNN': '#9B59B6'}

    for model_name, data in models_data.items():
        model_names.append(model_name)

        if 'folds' in data:
            # Calculate NPV for each fold
            fold_npvs = []
            for fold in data['folds']:
                metrics = calculate_metrics(fold['y_true'], fold['y_proba'], threshold)
                fold_npvs.append(metrics['NPV'])

            npv_mean = np.mean(fold_npvs)
            npv_std = np.std(fold_npvs)
            npv_means.append(npv_mean)
            npv_errors.append(npv_std)
        else:
            # Single model
            metrics = calculate_metrics(data['y_true'], data['y_proba'], threshold)
            npv_means.append(metrics['NPV'])
            npv_errors.append(0)

        colors_list.append(colors.get(model_name, 'gray'))

    # Create bar plot
    x_pos = np.arange(len(model_names))
    bars = ax.bar(x_pos, npv_means, yerr=npv_errors, capsize=5,
                   color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for i, (bar, npv, err) in enumerate(zip(bars, npv_means, npv_errors)):
        height = bar.get_height()
        if err > 0:
            label_text = f'{npv:.1%}\n±{err:.1%}'
        else:
            label_text = f'{npv:.1%}'
        ax.text(bar.get_x() + bar.get_width()/2., height + err + 0.01,
                label_text, ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Add interpretive text below
        missed_rate = 1 - npv
        ax.text(bar.get_x() + bar.get_width()/2., -0.08,
                f'{missed_rate:.1%} missed\nMSI-H cases',
                ha='center', va='top', fontsize=9, style='italic', color='#C0392B')

    ax.set_ylabel('Negative Predictive Value (NPV)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title(f'NPV Comparison Across Models (Threshold = {threshold})',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim([0, 1.05])
    ax.axhline(y=0.99, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Target NPV > 0.99')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved NPV comparison to {output_path}")
    plt.close()


def plot_npv_vs_threshold(models_data, output_path='npv_vs_threshold.png'):
    """
    Plot NPV vs threshold curves for all models.

    Args:
        models_data: Dictionary mapping model names to their data
        output_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = {'HistoBistro': '#E74C3C', 'CTransPath': '#3498DB', 'H-Optimus-0': '#2ECC71', 'CNN': '#9B59B6'}

    for model_name, data in models_data.items():
        if 'folds' in data:
            # Calculate for aggregated data
            agg = data['aggregated']
            thresholds, npvs, sens, specs = calculate_npv_vs_threshold(
                agg['y_true'], agg['y_proba']
            )
        else:
            thresholds, npvs, sens, specs = calculate_npv_vs_threshold(
                data['y_true'], data['y_proba']
            )

        color = colors.get(model_name, 'gray')

        # NPV vs Threshold
        ax1.plot(thresholds, npvs, color=color, linewidth=2.5, label=model_name)

        # Sensitivity vs NPV trade-off
        ax2.plot(sens, npvs, color=color, linewidth=2.5, label=model_name)

    # NPV vs Threshold plot
    ax1.axhline(y=0.99, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Target NPV = 0.99')
    ax1.axvline(x=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='Threshold = 0.5')
    ax1.set_xlabel('Decision Threshold', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Negative Predictive Value (NPV)', fontsize=12, fontweight='bold')
    ax1.set_title('NPV vs Decision Threshold', fontsize=13, fontweight='bold', pad=15)
    ax1.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.05])

    # Sensitivity vs NPV trade-off plot
    ax2.set_xlabel('Sensitivity (True Positive Rate)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Negative Predictive Value (NPV)', fontsize=12, fontweight='bold')
    ax2.set_title('Sensitivity vs NPV Trade-off', fontsize=13, fontweight='bold', pad=15)
    ax2.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved NPV vs threshold plot to {output_path}")
    plt.close()


def plot_performance_metrics(models_data, threshold=0.5, output_path='performance_metrics.png'):
    """
    Plot comprehensive performance metrics comparison.

    Args:
        models_data: Dictionary mapping model names to their data
        threshold: Decision threshold
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    metrics_names = ['AUROC', 'Sensitivity', 'Specificity', 'PPV', 'NPV']
    model_names = list(models_data.keys())
    x = np.arange(len(metrics_names))
    width = 0.25

    colors = {'HistoBistro': '#E74C3C', 'CTransPath': '#3498DB', 'H-Optimus-0': '#2ECC71', 'CNN': '#9B59B6'}

    for i, (model_name, data) in enumerate(models_data.items()):
        if 'folds' in data:
            # Calculate metrics for each fold
            fold_metrics = []
            for fold in data['folds']:
                m = calculate_metrics(fold['y_true'], fold['y_proba'], threshold)
                fold_metrics.append(m)

            # Average across folds
            means = [np.mean([fm[metric] for fm in fold_metrics]) for metric in metrics_names]
            stds = [np.std([fm[metric] for fm in fold_metrics]) for metric in metrics_names]
        else:
            m = calculate_metrics(data['y_true'], data['y_proba'], threshold)
            means = [m[metric] for metric in metrics_names]
            stds = [0] * len(metrics_names)

        offset = width * (i - len(model_names)/2 + 0.5)
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=4,
                      label=model_name, color=colors.get(model_name, 'gray'),
                      alpha=0.8, edgecolor='black', linewidth=1)

        # Add value labels
        for j, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                   f'{mean:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_title('Comprehensive Performance Metrics Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.axhline(y=0.99, color='green', linestyle='--', linewidth=2, alpha=0.3, label='Target = 0.99')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved performance metrics to {output_path}")
    plt.close()


def plot_clinical_impact(models_data, threshold=0.5, output_path='clinical_impact.png'):
    """
    Plot clinical impact visualization showing missed cases.

    Args:
        models_data: Dictionary mapping model names to their data
        threshold: Decision threshold
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(len(models_data), 1, figsize=(12, 4*len(models_data)))

    if len(models_data) == 1:
        axes = [axes]

    colors = {'HistoBistro': '#E74C3C', 'CTransPath': '#3498DB', 'H-Optimus-0': '#2ECC71', 'CNN': '#9B59B6'}

    for ax, (model_name, data) in zip(axes, models_data.items()):
        if 'folds' in data:
            agg = data['aggregated']
            y_true = agg['y_true']
            y_proba = agg['y_proba']
        else:
            y_true = data['y_true']
            y_proba = data['y_proba']

        metrics = calculate_metrics(y_true, y_proba, threshold)

        # Create icon grid (100 icons representing patients predicted negative)
        total_neg_calls = metrics['TN'] + metrics['FN']
        if total_neg_calls > 0:
            miss_rate = metrics['FN'] / total_neg_calls
        else:
            miss_rate = 0

        n_icons = 100
        n_missed = int(miss_rate * n_icons)
        n_correct = n_icons - n_missed

        # Create grid
        icons_per_row = 20
        grid = np.zeros((n_icons // icons_per_row, icons_per_row))

        # Fill with missed cases (randomly distributed)
        missed_indices = np.random.choice(n_icons, n_missed, replace=False)
        flat_grid = grid.flatten()
        flat_grid[missed_indices] = 1
        grid = flat_grid.reshape(grid.shape)

        # Plot
        im = ax.imshow(grid, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)

        ax.set_title(f'{model_name}\nOut of 100 patients called MSS: '
                    f'{n_correct} truly MSS (green), {n_missed} actually MSI-H (red)',
                    fontsize=13, fontweight='bold', pad=15)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Add text annotation
        ax.text(0.5, -0.15, f'NPV = {metrics["NPV"]:.1%}  |  Missed MSI-H rate = {miss_rate:.1%}',
               ha='center', va='top', transform=ax.transAxes,
               fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor=colors.get(model_name, 'gray'), alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved clinical impact visualization to {output_path}")
    plt.close()


def plot_score_distributions(models_data, threshold=0.5, output_path='score_distributions.png'):
    """
    Plot prediction score distributions for MSS vs MSI-H patients.

    Shows how well each model separates the two classes.

    Args:
        models_data: Dictionary mapping model names to their data
        threshold: Decision threshold to show as vertical line
        output_path: Path to save figure
    """
    n_models = len(models_data)
    fig, axes = plt.subplots(n_models, 1, figsize=(12, 4*n_models))

    if n_models == 1:
        axes = [axes]

    colors = {'HistoBistro': '#E74C3C', 'CTransPath': '#3498DB', 'H-Optimus-0': '#2ECC71', 'CNN': '#9B59B6'}

    # First pass: calculate histogram counts to determine global max for y-axis
    bins = np.linspace(0, 1, 41)  # 40 bins from 0 to 1
    global_max = 0

    for model_name, data in models_data.items():
        # Get aggregated predictions
        if 'folds' in data:
            agg = data['aggregated']
            y_true = agg['y_true']
            y_proba = agg['y_proba']
        else:
            y_true = data['y_true']
            y_proba = data['y_proba']

        mss_scores = y_proba[y_true == 0]
        msih_scores = y_proba[y_true == 1]

        # Get histogram counts
        mss_counts, _ = np.histogram(mss_scores, bins=bins)
        msih_counts, _ = np.histogram(msih_scores, bins=bins)

        # Update global max
        global_max = max(global_max, mss_counts.max(), msih_counts.max())

    # Add 10% padding to the max
    y_limit = global_max * 1.1

    for ax, (model_name, data) in zip(axes, models_data.items()):
        # Get aggregated predictions
        if 'folds' in data:
            agg = data['aggregated']
            y_true = agg['y_true']
            y_proba = agg['y_proba']
        else:
            y_true = data['y_true']
            y_proba = data['y_proba']

        # Separate scores by true class
        mss_scores = y_proba[y_true == 0]  # True negatives (MSS)
        msih_scores = y_proba[y_true == 1]  # True positives (MSI-H)

        # Plot MSS distribution (should be on left/low scores)
        ax.hist(mss_scores, bins=bins, alpha=0.6, color='steelblue',
                label=f'MSS (n={len(mss_scores)})', edgecolor='black', linewidth=0.5)

        # Plot MSI-H distribution (should be on right/high scores)
        ax.hist(msih_scores, bins=bins, alpha=0.6, color='firebrick',
                label=f'MSI-H (n={len(msih_scores)})', edgecolor='black', linewidth=0.5)

        # Add threshold line
        ax.axvline(x=threshold, color='black', linestyle='--', linewidth=2.5,
                   label=f'Threshold = {threshold}', zorder=10)

        # Add shaded regions
        ax.axvspan(0, threshold, alpha=0.1, color='steelblue', label='Predicted MSS\n(skip follow-up)')
        ax.axvspan(threshold, 1, alpha=0.1, color='firebrick', label='Predicted MSI-H\n(follow-up)')

        # Calculate metrics at this threshold
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        # Annotations
        ax.text(0.02, 0.98, f'NPV = {npv:.1%}\nMissed MSI-H: {fn}/{tn+fn}',
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlabel('Prediction Score (Probability of MSI-H)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Patients', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name} - Score Distribution by True Class',
                     fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=9)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, y_limit])  # Set standardized y-axis

        # Add text explaining the regions
        ax.text(threshold/2, y_limit*0.5, '← Predicted MSS\n(Send Home)',
                ha='center', va='center', fontsize=10, style='italic',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax.text((1+threshold)/2, y_limit*0.5, 'Predicted MSI-H →\n(Follow-up)',
                ha='center', va='center', fontsize=10, style='italic',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved score distribution plot to {output_path}")
    plt.close()


def print_summary_table(models_data, threshold=0.5):
    """
    Print summary table of metrics.

    Args:
        models_data: Dictionary mapping model names to their data
        threshold: Decision threshold
    """
    print("\n" + "="*80)
    print(f"PERFORMANCE SUMMARY (Threshold = {threshold})")
    print("="*80)
    print(f"{'Model':<20} {'AUROC':<12} {'Sensitivity':<12} {'Specificity':<12} {'NPV':<12} {'PPV':<12}")
    print("-"*80)

    for model_name, data in models_data.items():
        if 'folds' in data:
            # Calculate average across folds
            fold_metrics = []
            for fold in data['folds']:
                m = calculate_metrics(fold['y_true'], fold['y_proba'], threshold)
                fold_metrics.append(m)

            auroc_mean = np.mean([fm['AUROC'] for fm in fold_metrics])
            auroc_std = np.std([fm['AUROC'] for fm in fold_metrics])
            sens_mean = np.mean([fm['Sensitivity'] for fm in fold_metrics])
            spec_mean = np.mean([fm['Specificity'] for fm in fold_metrics])
            npv_mean = np.mean([fm['NPV'] for fm in fold_metrics])
            npv_std = np.std([fm['NPV'] for fm in fold_metrics])
            ppv_mean = np.mean([fm['PPV'] for fm in fold_metrics])

            print(f"{model_name:<20} {auroc_mean:.3f}±{auroc_std:.3f}  "
                  f"{sens_mean:.3f}       {spec_mean:.3f}       "
                  f"{npv_mean:.3f}±{npv_std:.3f}  {ppv_mean:.3f}")
        else:
            m = calculate_metrics(data['y_true'], data['y_proba'], threshold)
            print(f"{model_name:<20} {m['AUROC']:.3f}       "
                  f"{m['Sensitivity']:.3f}       {m['Specificity']:.3f}       "
                  f"{m['NPV']:.3f}       {m['PPV']:.3f}")

    print("="*80 + "\n")


def main():
    """Main function to generate all figures."""
    print("="*80)
    print("AORTIC MSI Prediction - Figure Generation")
    print("="*80)

    # Load all model data
    print("\nLoading data...")

    ctranspath_folds = load_stamp_predictions('ctranspath', n_folds=3)
    hoptimus_folds = load_stamp_predictions('h-optimus-0', n_folds=3)
    histobistro_data = load_histobistro_predictions()
    cnn_models = load_cnn_predictions()  # Returns list of 2 models (TCGA, DACHS)

    print(f"  - CTransPath: {len(ctranspath_folds)} folds loaded")
    print(f"  - H-Optimus-0: {len(hoptimus_folds)} folds loaded")
    print(f"  - HistoBistro: {len(histobistro_data['y_true'])} patients loaded")
    print(f"  - CNN: {len(cnn_models)} models loaded (TCGA, DACHS)")
    for cnn in cnn_models:
        print(f"    - CNN-{cnn['model_name']}: {len(cnn['y_true'])} patients")

    # Aggregate CV folds
    ctranspath_agg = aggregate_cv_folds(ctranspath_folds)
    hoptimus_agg = aggregate_cv_folds(hoptimus_folds)
    # For CNN, we can aggregate but keep individual runs visible too
    cnn_agg = aggregate_cv_folds(cnn_models)

    print(f"  - CTransPath aggregated: {len(ctranspath_agg['y_true'])} patients")
    print(f"  - H-Optimus-0 aggregated: {len(hoptimus_agg['y_true'])} patients")
    print(f"  - CNN aggregated: {len(cnn_agg['y_true'])} patients")

    # Organize data
    models_data = {
        'CNN': {
            'folds': cnn_models,  # TCGA and DACHS as separate runs
            'aggregated': cnn_agg
        },
        'HistoBistro': histobistro_data,
        'CTransPath': {
            'folds': ctranspath_folds,
            'aggregated': ctranspath_agg
        },
        'H-Optimus-0': {
            'folds': hoptimus_folds,
            'aggregated': hoptimus_agg
        }
    }

    # Print summary table
    print_summary_table(models_data, threshold=0.5)

    # Generate figures
    print("Generating figures...")

    plot_roc_curves(models_data, output_path='figure_1_roc_curves.png')
    plot_npv_comparison(models_data, threshold=0.5, output_path='figure_2_npv_comparison.png')
    plot_npv_vs_threshold(models_data, output_path='figure_3_npv_vs_threshold.png')
    plot_performance_metrics(models_data, threshold=0.5, output_path='figure_4_performance_metrics.png')
    plot_clinical_impact(models_data, threshold=0.5, output_path='figure_5_clinical_impact.png')
    plot_score_distributions(models_data, threshold=0.5, output_path='figure_6_score_distributions.png')

    print("\n" + "="*80)
    print("All figures generated successfully!")
    print("="*80)
    print("\nOutput files:")
    print("  - figure_1_roc_curves.png")
    print("  - figure_2_npv_comparison.png")
    print("  - figure_3_npv_vs_threshold.png")
    print("  - figure_4_performance_metrics.png")
    print("  - figure_5_clinical_impact.png")
    print("\n")


if __name__ == "__main__":
    main()
