#!/usr/bin/env python3
"""
Update all STAMP config files to use new folder structure.

This script updates:
- preprocessing.output_dir → results/stage3_features/{MODEL}/{SITE}/
- preprocessing.cache_dir → data/{SITE}/.cache
- crossval.output_dir → results/stage6_training/{MODEL}/{SITE}/crossval
- training.output_dir → results/stage6_training/{MODEL}/{SITE}/training
- statistics.output_dir → results/stage7_statistics/{MODEL}/{SITE}/
- heatmaps.output_dir → results/stage8_visualization/{MODEL}/{SITE}/heatmaps

Usage:
    python scripts/update_stamp_configs.py [--dry-run]
"""

import argparse
import yaml
from pathlib import Path
import re


BASE_DIR = Path("/lab/barcheese01/mdiberna/ARGO-DeepMSI")


def extract_site_from_config_name(config_path: Path) -> str:
    """Extract site name from config filename (e.g., config_OAUTHC.yaml → OAUTHC)."""
    match = re.search(r'config_(\w+)\.yaml', config_path.name)
    if match:
        return match.group(1)
    return None


def extract_model_from_path(config_path: Path) -> str:
    """Extract model name from config path (e.g., configs/ctranspath/... → ctranspath)."""
    return config_path.parent.name


def update_config_paths(config_path: Path, dry_run: bool = False) -> bool:
    """Update a single config file to use new folder structure.

    Args:
        config_path: Path to config YAML file
        dry_run: If True, print changes without writing

    Returns:
        True if config was updated, False otherwise
    """
    site = extract_site_from_config_name(config_path)
    model = extract_model_from_path(config_path)

    if not site or not model:
        print(f"⚠️  Skipping {config_path}: Could not extract site or model")
        return False

    # Skip config_all.yaml (consolidated configs)
    if site == "all":
        print(f"⚠️  Skipping {config_path}: Consolidated config (all sites)")
        return False

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Processing: {config_path.relative_to(BASE_DIR)}")
    print(f"  Model: {model}, Site: {site}")

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    updated = False

    # Update preprocessing section
    if 'preprocessing' in config:
        old_output = config['preprocessing'].get('output_dir')
        new_output = str(BASE_DIR / "results" / f"stage3_features" / model / site)

        if old_output != new_output:
            print(f"  preprocessing.output_dir:")
            print(f"    OLD: {old_output}")
            print(f"    NEW: {new_output}")
            config['preprocessing']['output_dir'] = new_output
            updated = True

        # Update cache_dir to use data/{SITE}/.cache
        old_cache = config['preprocessing'].get('cache_dir')
        new_cache = str(BASE_DIR / "data" / site / ".cache")

        if old_cache and old_cache != new_cache:
            print(f"  preprocessing.cache_dir:")
            print(f"    OLD: {old_cache}")
            print(f"    NEW: {new_cache}")
            config['preprocessing']['cache_dir'] = new_cache
            updated = True

    # Update crossval section
    if 'crossval' in config:
        # Update output_dir
        old_output = config['crossval'].get('output_dir')
        new_output = str(BASE_DIR / "results" / f"stage6_training" / model / site / "crossval")

        if old_output != new_output:
            print(f"  crossval.output_dir:")
            print(f"    OLD: {old_output}")
            print(f"    NEW: {new_output}")
            config['crossval']['output_dir'] = new_output
            updated = True

        # Update feature_dir to match new preprocessing output
        old_feature = config['crossval'].get('feature_dir')
        new_feature = str(BASE_DIR / "results" / f"stage3_features" / model / site)

        if old_feature != new_feature:
            print(f"  crossval.feature_dir:")
            print(f"    OLD: {old_feature}")
            print(f"    NEW: {new_feature}")
            config['crossval']['feature_dir'] = new_feature
            updated = True

    # Update training section
    if 'training' in config:
        old_output = config['training'].get('output_dir')
        new_output = str(BASE_DIR / "results" / f"stage6_training" / model / site / "training")

        if old_output != new_output:
            print(f"  training.output_dir:")
            print(f"    OLD: {old_output}")
            print(f"    NEW: {new_output}")
            config['training']['output_dir'] = new_output
            updated = True

        # Update feature_dir
        old_feature = config['training'].get('feature_dir')
        new_feature = str(BASE_DIR / "results" / f"stage3_features" / model / site)

        if old_feature != new_feature:
            config['training']['feature_dir'] = new_feature
            updated = True

    # Update statistics section
    if 'statistics' in config:
        old_output = config['statistics'].get('output_dir')
        new_output = str(BASE_DIR / "results" / f"stage7_statistics" / model / site)

        if old_output != new_output:
            print(f"  statistics.output_dir:")
            print(f"    OLD: {old_output}")
            print(f"    NEW: {new_output}")
            config['statistics']['output_dir'] = new_output
            updated = True

    # Update heatmaps section
    if 'heatmaps' in config:
        old_output = config['heatmaps'].get('output_dir')
        new_output = str(BASE_DIR / "results" / f"stage8_visualization" / model / site / "heatmaps")

        if old_output != new_output:
            print(f"  heatmaps.output_dir:")
            print(f"    OLD: {old_output}")
            print(f"    NEW: {new_output}")
            config['heatmaps']['output_dir'] = new_output
            updated = True

        # Update feature_dir
        old_feature = config['heatmaps'].get('feature_dir')
        new_feature = str(BASE_DIR / "results" / f"stage3_features" / model / site)

        if old_feature != new_feature:
            config['heatmaps']['feature_dir'] = new_feature
            updated = True

    # Write updated config
    if updated and not dry_run:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"  ✓ Updated config file")
    elif updated and dry_run:
        print(f"  [DRY RUN] Would update config file")
    else:
        print(f"  ✓ No changes needed")

    return updated


def main():
    parser = argparse.ArgumentParser(
        description="Update all STAMP config files to use new folder structure"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without modifying files"
    )
    args = parser.parse_args()

    # Find all config files
    configs_dir = BASE_DIR / "configs"
    config_files = list(configs_dir.glob("*/config_*.yaml"))

    print(f"Found {len(config_files)} config files to process")
    print("=" * 80)

    updated_count = 0
    for config_path in sorted(config_files):
        if update_config_paths(config_path, dry_run=args.dry_run):
            updated_count += 1

    print("\n" + "=" * 80)
    print(f"Summary: {updated_count}/{len(config_files)} configs updated")

    if args.dry_run:
        print("\n[DRY RUN] No files were modified. Run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
