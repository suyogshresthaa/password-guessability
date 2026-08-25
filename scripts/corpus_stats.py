"""Step 1 demo: run the real pipeline over the real 14.3M-line corpus.

Prints exactly what data.py measures — nothing here is a toy-scale number.
"""

import time
from pathlib import Path

from pwguess.data import corpus_stats, load_eval_passwords, overlap_with_eval

ROOT = Path(__file__).resolve().parents[1]
ROCKYOU = ROOT / "data" / "raw" / "rockyou.txt"
EVAL_DB = ROOT / "data" / "eval" / "000webhost_100k.sqlite"


def main():
    t0 = time.time()
    stats, hashes = corpus_stats(ROCKYOU)
    t1 = time.time()

    print("=== RockYou corpus ===")
    print(f"  total passwords     : {stats['total']:>12,}")
    print(f"  train (98%)         : {stats['n_train']:>12,}")
    print(f"  val   (2%)          : {stats['n_val']:>12,}")
    print(f"  contain an out-of-vocab char : {stats['with_oov_chars']:>10,}  ({stats['with_oov_frac']:.4%})")
    print(f"  longer than MAX_LEN-2 (truncated) : {stats['truncated']:>6,}  ({stats['truncated_frac']:.4%})")
    print(f"  scanned in {t1 - t0:.1f}s")

    eval_pw = load_eval_passwords(EVAL_DB)
    overlap = overlap_with_eval(hashes, eval_pw)
    t2 = time.time()

    print("\n=== Overlap: data/eval/000webhost_100k.sqlite vs RockYou ===")
    print(f"  eval set size       : {overlap['n_eval']:>12,}")
    print(f"  verbatim in RockYou : {overlap['n_overlap']:>12,}  ({overlap['overlap_frac']:.2%})")
    print(f"  overlap check in {t2 - t1:.1f}s")
    print()
    print("  This is reported, not removed: a real attacker has RockYou too,")
    print("  so this overlap is genuine attack surface, not a data leak.")
    print("  Phase 3 evaluates on the full set AND on the non-overlapping")
    print("  subset separately, per docs/00-audit-v1.md.")


if __name__ == "__main__":
    main()
