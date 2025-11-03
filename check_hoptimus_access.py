#!/usr/bin/env python3
"""
Check access to H-optimus models
"""

from huggingface_hub import HfApi

def check_model_access(model_name):
    """Check if we have access to a specific model."""
    try:
        api = HfApi()
        model_info = api.model_info(model_name)
        print(f"✓ {model_name}: Access confirmed")
        return True
    except Exception as e:
        print(f"✗ {model_name}: {e}")
        return False

def main():
    print("Checking H-optimus model access...")
    
    # Check login status first
    try:
        api = HfApi()
        user = api.whoami()
        print(f"✓ Logged in as: {user['name']}")
    except Exception as e:
        print(f"✗ Not logged in: {e}")
        print("Please run: hf auth login")
        return
    
    print("\nChecking model access:")
    
    # Check both H-optimus models
    h0_access = check_model_access("bioptimus/H-optimus-0")
    h1_access = check_model_access("bioptimus/H-optimus-1")
    
    print(f"\nSummary:")
    if h0_access:
        print("✓ H-optimus-0 is available - you can use this model")
    if h1_access:
        print("✓ H-optimus-1 is available - you can use this model")
    
    if not h0_access and not h1_access:
        print("✗ No H-optimus models accessible")
        print("Please request access at:")
        print("  - https://huggingface.co/bioptimus/H-optimus-0")
        print("  - https://huggingface.co/bioptimus/H-optimus-1")
    elif h0_access and not h1_access:
        print("Recommendation: Use H-optimus-0 while waiting for H-optimus-1 access")

if __name__ == "__main__":
    main()