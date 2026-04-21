#!/bin/bash
# Convert non-pyramidal slides to pyramidal TIFFs for LazySlide compatibility.
#
# Problem: LazySlide's find_tissues() loads the thumbnail level. Non-pyramidal
# slides (n_levels=1) force loading the full resolution image, causing OOMs
# on large slides (e.g., 78K×75K @ 256G).
#
# Solution: Convert to pyramidal TIFF with vips. Fast (~1-2 min per slide),
# preserves image quality, and the resulting TIFF works with OpenSlide.
#
# Usage:
#   bash scripts/convert_pyramidal.sh results/data/slide_table.csv
#
# Requirements:
#   - vips (libvips-tools): conda install -c conda-forge libvips
#   - Python with openslide-python (already in argo env)
#
# What it does:
#   1. Reads FILENAME column from slide_table.csv
#   2. Checks each slide for pyramid levels
#   3. Non-pyramidal slides: converts to .pyramidal.tiff alongside original
#   4. Writes an updated slide table with converted paths

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <slide_table.csv> [--dry-run]"
    exit 1
fi

SLIDE_TABLE="$1"
DRY_RUN="${2:-}"
CONVERTED=0
SKIPPED=0
FAILED=0

echo "========================================"
echo "ARGO-DeepMSI: Pyramidal Conversion"
echo "========================================"
echo "Slide table: $SLIDE_TABLE"
echo "Slides: $(tail -n +2 "$SLIDE_TABLE" | wc -l)"
echo ""

# Check vips is available
if ! command -v vips &> /dev/null; then
    echo "ERROR: vips not found. Install with:"
    echo "  conda install -c conda-forge libvips"
    echo "  # or: apt install libvips-tools"
    exit 1
fi

# Create output table
OUTPUT_TABLE="${SLIDE_TABLE%.csv}_pyramidal.csv"
head -n 1 "$SLIDE_TABLE" > "$OUTPUT_TABLE"

# Process each slide
tail -n +2 "$SLIDE_TABLE" | while IFS=, read -r line; do
    # Extract FILENAME (second column after PATIENT, or find it)
    SLIDE_PATH=$(echo "$line" | python3 -c "
import sys, csv
reader = csv.reader(sys.stdin)
row = next(reader)
# Find FILENAME column by reading header
import io
header = open('$SLIDE_TABLE').readline().strip().split(',')
idx = header.index('FILENAME')
print(row[idx])
" 2>/dev/null)

    if [ -z "$SLIDE_PATH" ] || [ ! -f "$SLIDE_PATH" ]; then
        echo "  SKIP (not found): $SLIDE_PATH"
        echo "$line" >> "$OUTPUT_TABLE"
        ((SKIPPED++)) || true
        continue
    fi

    SLIDE_NAME=$(basename "$SLIDE_PATH")

    # Check pyramid levels
    N_LEVELS=$(python3 -c "
try:
    import openslide
    s = openslide.OpenSlide('$SLIDE_PATH')
    print(s.level_count)
except Exception as e:
    print(f'ERROR:{e}')
" 2>/dev/null)

    if [[ "$N_LEVELS" == ERROR* ]]; then
        echo "  SKIP (can't open): $SLIDE_NAME — $N_LEVELS"
        echo "$line" >> "$OUTPUT_TABLE"
        ((SKIPPED++)) || true
        continue
    fi

    if [ "$N_LEVELS" -gt 1 ]; then
        echo "  OK ($N_LEVELS levels): $SLIDE_NAME"
        echo "$line" >> "$OUTPUT_TABLE"
        ((SKIPPED++)) || true
        continue
    fi

    # Non-pyramidal — needs conversion
    CONVERTED_PATH="${SLIDE_PATH%.svs}.pyramidal.tiff"

    if [ -f "$CONVERTED_PATH" ]; then
        echo "  ALREADY CONVERTED: $SLIDE_NAME"
        # Update path in output
        echo "$line" | sed "s|$SLIDE_PATH|$CONVERTED_PATH|g" >> "$OUTPUT_TABLE"
        ((SKIPPED++)) || true
        continue
    fi

    if [ "$DRY_RUN" = "--dry-run" ]; then
        echo "  WOULD CONVERT (1 level): $SLIDE_NAME"
        echo "$line" >> "$OUTPUT_TABLE"
        continue
    fi

    echo "  CONVERTING (1 level): $SLIDE_NAME"
    DIMS=$(python3 -c "
import openslide
s = openslide.OpenSlide('$SLIDE_PATH')
print(f'{s.dimensions[0]}x{s.dimensions[1]}')
")
    echo "    Dimensions: $DIMS"

    if vips tiffsave "$SLIDE_PATH" "$CONVERTED_PATH" \
        --pyramid \
        --tile \
        --tile-width 256 \
        --tile-height 256 \
        --compression jpeg \
        --Q 90 \
        2>/dev/null; then

        echo "    ✓ Converted: $(basename $CONVERTED_PATH)"
        # Update path in output table
        echo "$line" | sed "s|$SLIDE_PATH|$CONVERTED_PATH|g" >> "$OUTPUT_TABLE"
        ((CONVERTED++)) || true
    else
        echo "    ✗ FAILED to convert: $SLIDE_NAME"
        echo "$line" >> "$OUTPUT_TABLE"
        ((FAILED++)) || true
    fi
done

echo ""
echo "========================================"
echo "Results: $CONVERTED converted, $SKIPPED skipped, $FAILED failed"
echo "Updated slide table: $OUTPUT_TABLE"
echo "========================================"
