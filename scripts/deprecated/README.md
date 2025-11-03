# Deprecated Scripts

These scripts are from the previous version of ARGO-DeepMSI before the refactoring to use the installable `argo-deepmsi` package.

**DO NOT USE THESE SCRIPTS.** They are kept for reference only.

## Old Scripts → New Pipeline

| Old Script | Replaced By | Status |
|-----------|-------------|--------|
| `0.prepare.py` | `scripts/1_data_ingestion.py` | ✅ Logic moved to `argo_deepmsi.data_ingestion` |
| `2.preprocess_eval.py` | `scripts/4_feature_validation.py` | ✅ Logic moved to `argo_deepmsi.feature_validation` |
| `2.preprocess_eval-h-optimus-0.py` | `scripts/4_feature_validation.py --extractor h-optimus-0` | ✅ Unified into single script with extractor parameter |
| `6.prepare_histobistro.py` | `scripts/5_baseline_testing.py` | ⏳ To be implemented in `argo_deepmsi.baseline_testing` |
| `5.embedding_visualizations_h-optimus-0.py` | `scripts/8_visualization.py` | ⏳ To be implemented in `argo_deepmsi.visualization` |

## Why These Were Deprecated

1. **Heavy logic in scripts**: The old scripts contained 500+ lines of business logic mixed with CLI code
2. **No reusability**: Functions couldn't be imported or used elsewhere
3. **Inconsistent patterns**: Each script had its own style and structure
4. **Hard to test**: No separation between logic and I/O
5. **Not extensible**: Difficult to add custom models or modifications

## New Architecture

The new pipeline uses:
- **`argo_deepmsi/`**: Installable Python package with all core logic
- **`scripts/`**: Thin CLI wrappers (50-100 lines) that call package functions
- **Clear stages**: Numbered pipeline scripts (1-8) for each stage
- **Logging**: Consistent logging to `logs/` directory
- **Type hints**: Better code documentation and IDE support

See the main README.md for the new pipeline structure and usage.
