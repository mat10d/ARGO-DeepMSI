#!/usr/bin/env python3
"""
Test STAMP model access and availability.

This script checks if a model can be loaded before running feature extraction.
For Hugging Face gated models, it verifies authentication.

Usage:
    python scripts/test_model_access.py ctranspath
    python scripts/test_model_access.py h-optimus-0
    python scripts/test_model_access.py virchow2

Exit codes:
    0 - Model is accessible
    1 - Model is not accessible (auth failure, missing, etc.)
    2 - STAMP not available
"""

import sys
import os
from pathlib import Path


# Set HF cache directory
os.environ["HF_HOME"] = "/lab/barcheese01/mdiberna/ARGO-DeepMSI/.huggingface_cache"
os.environ["HF_DATASETS_CACHE"] = os.path.join(os.environ["HF_HOME"], "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.environ["HF_HOME"], "transformers")


def test_stamp_available() -> bool:
    """Check if STAMP is available."""
    try:
        import stamp
        print(f"✓ STAMP available")
        return True
    except ImportError:
        print(f"✗ STAMP not found. Activate STAMP environment first.")
        return False


def test_hf_authentication() -> bool:
    """Check if Hugging Face authentication is valid."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user = api.whoami()
        print(f"✓ Hugging Face authenticated as: {user['name']}")
        return True
    except Exception as e:
        print(f"✗ Hugging Face authentication failed: {e}")
        print(f"  Run: hf auth login")
        return False


def test_model_access(model: str) -> bool:
    """Test if a specific model can be loaded.

    Args:
        model: Model name (e.g., "ctranspath", "virchow2")

    Returns:
        True if model is accessible, False otherwise.
    """
    print(f"\nTesting model: {model}")
    print("=" * 60)

    # Models requiring HF authentication
    HF_GATED_MODELS = [
        "h-optimus-0",
        "h-optimus-1",
        "virchow2",
        "virchow",
        "uni2",
        "uni",
        "conch1_5",
        "conch",
        "gigapath",
        "mstar",
        "musk",
    ]

    # Check if HF auth is required and valid
    if model in HF_GATED_MODELS:
        if not test_hf_authentication():
            return False

    # Try to load the model via STAMP
    try:
        from stamp.modeling.extractors import get_extractor

        print(f"Loading {model} extractor...")
        extractor = get_extractor(model, device="cpu")  # Use CPU for testing
        print(f"✓ Model {model} loaded successfully")

        # Clean up
        del extractor

        return True

    except ImportError as e:
        print(f"✗ Failed to import STAMP extractor: {e}")
        return False
    except Exception as e:
        error_msg = str(e).lower()

        # Check for common errors
        if "authentication" in error_msg or "401" in error_msg or "403" in error_msg:
            print(f"✗ Authentication error for {model}")
            print(f"  This model requires Hugging Face access approval")
            print(f"  Visit: https://huggingface.co/models")
        elif "not found" in error_msg or "404" in error_msg:
            print(f"✗ Model {model} not found")
        elif "connection" in error_msg or "timeout" in error_msg:
            print(f"✗ Network error loading {model}: {e}")
        else:
            print(f"✗ Failed to load {model}: {e}")

        return False


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/test_model_access.py <MODEL>")
        print("Example: python scripts/test_model_access.py ctranspath")
        sys.exit(2)

    model = sys.argv[1]

    # Check STAMP availability first
    if not test_stamp_available():
        sys.exit(2)

    # Test model access
    if test_model_access(model):
        print(f"\n✓ Model {model} is accessible and ready to use")
        sys.exit(0)
    else:
        print(f"\n✗ Model {model} is not accessible")
        sys.exit(1)


if __name__ == "__main__":
    main()
