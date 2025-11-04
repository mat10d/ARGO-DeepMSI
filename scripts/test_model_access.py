#!/usr/bin/env python3
"""
Test STAMP model access and availability.

This script performs lightweight checks before running feature extraction:
- STAMP is installed
- Extractor file exists for the model
- For Hugging Face gated models, authentication is valid

Usage:
    python scripts/test_model_access.py ctranspath
    python scripts/test_model_access.py h-optimus-0
    python scripts/test_model_access.py virchow2

Exit codes:
    0 - Model is accessible (all prerequisites met)
    1 - Model is not accessible (auth failure, missing requirements, etc.)
    2 - STAMP not available
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

# Set HF cache directory
os.environ["HF_HOME"] = "/lab/barcheese01/mdiberna/ARGO-DeepMSI/.huggingface_cache"
os.environ["HF_DATASETS_CACHE"] = os.path.join(os.environ["HF_HOME"], "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.environ["HF_HOME"], "transformers")

# Set HF token from .env if available
if "HF_TOKEN" in os.environ and os.environ["HF_TOKEN"]:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]


def test_stamp_available() -> tuple[bool, Path | None]:
    """Check if STAMP is available and return its path."""
    try:
        import stamp
        stamp_path = Path(stamp.__file__).parent
        print(f"✓ STAMP available (version {stamp.__version__})")
        return True, stamp_path
    except ImportError:
        print(f"✗ STAMP not found. Activate STAMP environment first.")
        return False, None


def test_hf_authentication() -> bool:
    """Check if Hugging Face authentication is valid."""
    try:
        from huggingface_hub import HfApi

        # Get token from environment
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

        if not token:
            print(f"✗ Hugging Face token not found in environment")
            print(f"  Add HF_TOKEN to .env file or run: huggingface-cli login")
            return False

        api = HfApi(token=token)
        user = api.whoami(token=token)
        print(f"✓ Hugging Face authenticated as: {user['name']}")
        return True
    except Exception as e:
        print(f"✗ Hugging Face authentication failed: {e}")
        print(f"  Check your HF_TOKEN in .env file or run: huggingface-cli login")
        return False


def test_model_prerequisites(model: str, stamp_path: Path) -> bool:
    """Test if prerequisites for a specific model are met.

    Args:
        model: Model name (e.g., "ctranspath", "virchow2")
        stamp_path: Path to STAMP installation

    Returns:
        True if prerequisites are met, False otherwise.
    """
    print(f"\nTesting model: {model}")
    print("=" * 60)

    # Models requiring HF authentication
    HF_GATED_MODELS = [
        "h-optimus-0",
        "h-optimus-1",
        "virchow2",
        "virchow",
        "virchow-full",
        "uni2",
        "uni",
        "conch1_5",
        "conch",
        "gigapath",
        "mstar",
        "musk",
    ]

    # Map model names to their module file names (handle special cases)
    MODEL_TO_FILE = {
        "h-optimus-0": "h_optimus_0.py",
        "h-optimus-1": "h_optimus_1.py",
        "chief-ctranspath": "chief_ctranspath.py",
        "dino-bloom": "dinobloom.py",
        "virchow-full": "virchow_full.py",
    }

    # Check if HF auth is required and valid
    if model in HF_GATED_MODELS:
        print(f"Model {model} requires Hugging Face authentication")
        if not test_hf_authentication():
            return False

    # Check if the extractor file exists
    file_name = MODEL_TO_FILE.get(model, f"{model}.py")
    extractor_path = stamp_path / "preprocessing" / "extractor" / file_name

    if extractor_path.exists():
        print(f"✓ Extractor file found: {extractor_path.name}")
        return True
    else:
        print(f"✗ Extractor file not found: {extractor_path}")
        print(f"  This model may not be supported by this version of STAMP")
        return False


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/test_model_access.py <MODEL>")
        print("Example: python scripts/test_model_access.py ctranspath")
        sys.exit(2)

    model = sys.argv[1]

    # Check STAMP availability first
    stamp_available, stamp_path = test_stamp_available()
    if not stamp_available or stamp_path is None:
        sys.exit(2)

    # Test model prerequisites
    if test_model_prerequisites(model, stamp_path):
        print(f"\n✓ Model {model} prerequisites satisfied")
        sys.exit(0)
    else:
        print(f"\n✗ Model {model} is not accessible")
        sys.exit(1)


if __name__ == "__main__":
    main()
