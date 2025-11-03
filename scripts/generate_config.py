#!/usr/bin/env python3
"""
Generate STAMP config files from templates.

This script reads a template config file, replaces placeholders with actual values,
and writes the generated config to a specified output location.

Usage:
    python scripts/generate_config.py \\
        --template configs/templates/preprocessing_site.yaml.template \\
        --output .temp_configs/ctranspath/config_OAUTHC.yaml \\
        --model ctranspath \\
        --site OAUTHC \\
        --device cuda:0

Placeholders supported:
    ${BASE_DIR}  - Project base directory
    ${MODEL}     - Feature extractor model name
    ${SITE}      - Site name
    ${DEVICE}    - CUDA device
    ${N_SPLITS}  - Number of cross-validation folds
"""

import argparse
import re
from pathlib import Path
import sys


def replace_placeholders(content: str, replacements: dict) -> str:
    """Replace ${PLACEHOLDER} with actual values.

    Args:
        content: Template content with placeholders
        replacements: Dictionary mapping placeholder names to values

    Returns:
        Content with placeholders replaced
    """
    result = content

    for key, value in replacements.items():
        placeholder = f"${{{key}}}"
        result = result.replace(placeholder, str(value))

    # Check for unreplaced placeholders
    remaining = re.findall(r'\$\{(\w+)\}', result)
    if remaining:
        print(f"WARNING: Unreplaced placeholders: {', '.join(set(remaining))}", file=sys.stderr)

    return result


def generate_config(
    template_path: Path,
    output_path: Path,
    replacements: dict
) -> None:
    """Generate config file from template.

    Args:
        template_path: Path to template file
        output_path: Path to write generated config
        replacements: Dictionary of placeholder replacements
    """
    # Read template
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    with open(template_path, 'r') as f:
        template_content = f.read()

    # Replace placeholders
    config_content = replace_placeholders(template_content, replacements)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write generated config
    with open(output_path, 'w') as f:
        f.write(config_content)

    print(f"✓ Generated config: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate STAMP config from template",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument(
        "--template",
        type=Path,
        required=True,
        help="Path to template file (e.g., configs/templates/preprocessing_site.yaml.template)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write generated config (e.g., .temp_configs/ctranspath/config_OAUTHC.yaml)"
    )

    # Placeholder values
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/lab/barcheese01/mdiberna/ARGO-DeepMSI",
        help="Project base directory (default: current project)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Feature extractor model name (e.g., ctranspath, virchow2, h-optimus-0)"
    )
    parser.add_argument(
        "--site",
        type=str,
        help="Site name (e.g., OAUTHC, LUTH, LASUTH, all)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device (default: cuda:0)"
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=3,
        help="Number of cross-validation folds (default: 3)"
    )

    # Additional custom replacements
    parser.add_argument(
        "--set",
        action="append",
        metavar="KEY=VALUE",
        help="Set custom placeholder value (can be used multiple times)"
    )

    args = parser.parse_args()

    # Build replacements dictionary
    replacements = {
        "BASE_DIR": args.base_dir,
        "DEVICE": args.device,
        "N_SPLITS": args.n_splits,
    }

    if args.model:
        replacements["MODEL"] = args.model

    if args.site:
        replacements["SITE"] = args.site

    # Add custom replacements from --set
    if args.set:
        for item in args.set:
            if '=' not in item:
                print(f"ERROR: Invalid --set format: {item} (expected KEY=VALUE)", file=sys.stderr)
                sys.exit(1)
            key, value = item.split('=', 1)
            replacements[key] = value

    # Generate config
    try:
        generate_config(args.template, args.output, replacements)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
