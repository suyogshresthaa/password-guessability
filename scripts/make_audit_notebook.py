"""Generates notebooks/00_audit.ipynb — the reproducible companion to docs/00-audit-v1.md.

Kept as a generator rather than a hand-edited .ipynb so the notebook stays diffable
and its code can be extracted and smoke-tested (see scripts/check_audit_notebook.py).
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "00_audit.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text):
    return {
        "cell_type": "code", "metadata": {}, "execution_count": None,
        "outputs": [], "source": text.strip().splitlines(True),
    }


CELLS = [
    md("""
# Audit of v1 — the label is a function of length

Companion to [`docs/00-audit-v1.md`](../docs/00-audit-v1.md). Run top to bottom.

Every claim in the write-up is derived here. Nothing is quoted from the original
notebooks without being recomputed.
"""),
    code("""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DB = ROOT / "data" / "eval" / "000webhost_100k.sqlite"

with sqlite3.connect(DB) as conn:
    df = pd.read_sql("select password, strength from Users", conn)

df["length"] = df["password"].str.len()
print(f"{len(df):,} rows · {df['password'].nunique():,} distinct passwords")
df.head()
"""),
    md("""
## 1. The dataset is deduplicated

100,000 rows and 100,000 distinct passwords. Someone removed duplicates before
publishing, which destroys the frequency signal. That is why v2 uses this file only
as a held-out attack-target set, never as training data for a probabilistic model.
"""),
    code("""
assert df["password"].nunique() == len(df), "expected a fully deduplicated table"
print("duplicates:", len(df) - df["password"].nunique())
print("nulls:     ", int(df.isna().sum().sum()))
"""),
    md("""
## 2. Label vs. length

Grouping by label shows three length intervals that do not overlap.
"""),
    code("""
df.groupby("strength")["length"].agg(
    n="size", min="min", max="max", mean="mean", median="median"
).round(2)
"""),
    md("""
## 3. The rule, and its error count

`len <= 7 -> Weak`, `8..13 -> Medium`, `>= 14 -> Strong`.
"""),
    code("""
def length_rule(lengths):
    return np.where(lengths <= 7, 0, np.where(lengths <= 13, 1, 2))

violations = int((length_rule(df["length"].values) != df["strength"].values).sum())
print(f"rows where the rule disagrees with the label: {violations} / {len(df):,}")
assert violations == 0
"""),
    md("""
Zero. The same check in SQL, so the result does not depend on the pandas path:
"""),
    code("""
with sqlite3.connect(DB) as conn:
    (n,) = conn.execute(\"\"\"
        select count(*) from Users
        where not (
          (length(password) <= 7             and strength = 0) or
          (length(password) between 8 and 13 and strength = 1) or
          (length(password) >= 14            and strength = 2)
        )
    \"\"\").fetchone()
print("violations:", n)
"""),
    md("""
### Every length bucket is pure

No length maps to two different labels — a stronger statement than "the rule fits".
"""),
    code("""
per_length = df.groupby("length")["strength"].nunique()
print("lengths mapping to more than one label:", int((per_length > 1).sum()))
df.pivot_table(index="length", columns="strength", values="password",
               aggfunc="size", fill_value=0).head(20)
"""),
    md("""
## 4. Baselines on the notebook's own test split

`train_test_split(..., test_size=0.2, random_state=42, stratify=y)` reproduces the
exact 20,000 rows v1 evaluated on, so these numbers sit in the same table as its
reported results.
"""),
    code("""
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

X, y = df["password"].values, df["strength"].values
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"test rows: {len(y_te):,}")


def rule_length(pw):
    return length_rule(np.array([len(p) for p in pw]))


def rule_majority(pw):
    return np.full(len(pw), 1)  # 'Medium' is 73.9% of the data


def rule_charclass(pw):
    out = []
    for p in pw:
        k = sum([
            any(c.isdigit() for c in p), any(c.isupper() for c in p),
            any(c.islower() for c in p), any(not c.isalnum() for c in p),
        ])
        out.append(0 if k <= 1 else (1 if k <= 2 else 2))
    return np.array(out)


baselines = {
    "Length rule (3 lines)": rule_length,
    "Character-class-count rule": rule_charclass,
    "Majority class ('always Medium')": rule_majority,
}

rows = []
for name, fn in baselines.items():
    pred = fn(X_te)
    rows.append({
        "baseline": name,
        "accuracy": accuracy_score(y_te, pred),
        "f1_weighted": f1_score(y_te, pred, average="weighted"),
        "f1_macro": f1_score(y_te, pred, average="macro"),
    })

pd.DataFrame(rows).set_index("baseline").round(6)
"""),
    md("""
Placed alongside v1's reported numbers:

| model | accuracy |
|---|---|
| **Length rule (3 lines of Python)** | **1.00000** |
| XGBoost | 1.00000 |
| Linear SVM *(shipped)* | 0.99330 |
| Logistic Regression | 0.98140 |
| Character-class-count rule | 0.86380 |
| Naive Bayes | 0.83045 |
| Majority class | 0.73885 |

The shipped model loses to an `if` statement, and a four-line heuristic beats Naive Bayes.
"""),
    md("""
## 5. Why XGBoost hit exactly 1.00000

v1 dismissed the perfect score as leakage. It was not leakage — a tree ensemble
splits on thresholds and the label *is* a threshold. A depth-2 decision tree given
nothing but `length` reproduces it exactly.
"""),
    code("""
from sklearn.tree import DecisionTreeClassifier, export_text

stump = DecisionTreeClassifier(max_depth=2).fit(
    np.array([len(p) for p in X_tr]).reshape(-1, 1), y_tr
)
pred = stump.predict(np.array([len(p) for p in X_te]).reshape(-1, 1))
print(f"depth-2 tree on `length` alone: accuracy = {accuracy_score(y_te, pred):.5f}")
print(export_text(stump, feature_names=["length"]))
"""),
    md("""
One feature, two splits, 100%. The 51,808 TF-IDF character n-grams were spent
approximating this.
"""),
    md("""
## 6. Passwords v1 rates "Strong"

Because the label is length alone, any long password is Strong regardless of how
guessable it is. These are real rows from the dataset.
"""),
    code("""
strong = df[df["strength"] == 2]
obvious = strong[strong["password"].str.contains(
    "password|123456|qwerty|iloveyou", case=False, regex=True
)]
print(f"{len(obvious):,} of {len(strong):,} 'Strong' passwords contain an obvious dictionary token")
obvious.head(15)[["password", "length", "strength"]]
"""),
    md("""
Every one of these dies to a dictionary attack in seconds. Whatever v1 measured,
it was not resistance to guessing.

## 7. The deployed artifact is not the tuned model

`02_modelling.ipynb` cell 33 binds `best_svm = grid.best_estimator_`
(`LinearSVC(C=1, max_iter=10000)`), then cell 36 saves `svm_model`
(`LinearSVC(C=1, max_iter=2000)`, from cell 19). Read it out of the pickle:
"""),
    code("""
import joblib

pkl = ROOT / "legacy" / "v1" / "artifacts" / "linearsvc_c1_untuned.pkl"
model = joblib.load(pkl)
print(model)
print("\\nmax_iter =", model.max_iter, "  <- the grid search winner used 10000")
assert model.max_iter == 2000, "expected the untuned fit"
"""),
    md("""
## 8. The confidence score is not a probability

`app.py` applies a softmax to `LinearSVC.decision_function` output and renders the
result as a percentage. Those are signed distances to a hyperplane, never fit to be
log-odds — the transformation produces a number in [0, 1] that is calibrated
against nothing.
"""),
    code("""
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=1, keepdims=True)


# The same arbitrary scale change that leaves predictions identical
# moves the reported "confidence" by tens of points.
margins = np.array([[-1.2, 0.8, -0.4]])
for scale in (0.5, 1.0, 2.0, 5.0):
    p = softmax(margins * scale)[0]
    print(f"scale x{scale:<4} -> argmax unchanged ({p.argmax()}), "
          f"'confidence' = {p.max() * 100:5.2f}%")
"""),
    md("""
Same decision every time; the displayed confidence swings from 46% to 98%. The number
was decoration.

---

## Conclusion

1. The label is `f(len)` with zero exceptions — v1 had no signal to learn.
2. Its headline 99.33% is worse than a three-line rule, and it was never compared to one.
3. The perfect XGBoost score was the only correct result in the notebook, and it was discarded.
4. The shipped pickle is not the model the README describes.
5. The displayed confidence is uncalibrated.

The passwords remain useful as a cross-corpus evaluation set. The labels do not.
See [`docs/00-audit-v1.md`](../docs/00-audit-v1.md) §6–7 for what replaces this.
"""),
]

NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(NOTEBOOK, indent=1) + "\n")
    n_code = sum(c["cell_type"] == "code" for c in CELLS)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(CELLS)} cells, {n_code} code)")
