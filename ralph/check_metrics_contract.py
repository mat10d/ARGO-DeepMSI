"""Assert every scorer that has emitted results carries the FULL metric block.

A scorer's results/scorers/<name>/metrics.json must contain the screening block
keys (spec_at_sens90, spec_at_sens95, npv_at_sens95, auroc, auprc) plus by_site.
Scorers with no metrics.json yet (not run) are skipped — this only polices what
has been produced, so partial/AUROC-only landings fail the gate.

Exit 0 if all present metrics.json are complete; non-zero otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED = ["spec_at_sens90", "spec_at_sens95", "npv_at_sens95", "auroc", "auprc"]
RESULTS = Path("results/scorers")


def main() -> int:
    if not RESULTS.exists():
        print(f"[metrics_contract] {RESULTS} not present yet — nothing to check.")
        return 0
    bad = []
    checked = 0
    for mj in RESULTS.glob("*/metrics.json"):
        checked += 1
        try:
            d = json.loads(mj.read_text())
        except Exception as e:
            bad.append(f"{mj}: unreadable ({e})")
            continue
        missing = [k for k in REQUIRED if k not in d]
        if "by_site" not in d:
            missing.append("by_site")
        if missing:
            bad.append(f"{mj}: missing {missing}")
    if bad:
        print("[metrics_contract] FAIL:\n  " + "\n  ".join(bad))
        return 1
    print(f"[metrics_contract] OK: {checked} metrics.json complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
