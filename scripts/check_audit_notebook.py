"""Smoke-test notebooks/00_audit.ipynb by executing its code cells in order.

Cheaper than a jupyter dependency, and it fails loudly if the audit's claims stop
holding — every assertion in the notebook is a claim made in docs/00-audit-v1.md.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "00_audit.ipynb"


def main():
    cells = [
        "".join(c["source"])
        for c in json.loads(NB.read_text())["cells"]
        if c["cell_type"] == "code"
    ]
    ns = {"__name__": "__notebook__"}
    for i, src in enumerate(cells, 1):
        try:
            exec(compile(src, f"<cell {i}>", "exec"), ns)
        except Exception as exc:
            print(f"FAIL cell {i}: {type(exc).__name__}: {exc}", file=sys.stderr)
            print("-" * 60, file=sys.stderr)
            print(src, file=sys.stderr)
            return 1
        print(f"  cell {i}/{len(cells)} ok")
    print(f"\nall {len(cells)} code cells executed cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
