"""Learned fusion of the stage-1 score and the three cross-encoder scores (NumPy only).

Features per candidate (all computed within the query's 50 candidates, so model A scores on the
proxy splits and model B scores on test are on the same scale):
  z(stage 1), z(rerank), z(hardneg), z(aug), 1 / log2(stage-1 rank + 2)
Model: one linear score per candidate, trained with a listwise softmax cross-entropy (the correct
product against the other candidates of its query) plus an L2 penalty (lambda = 1, fixed), by
Newton's method from zero with a backtracking line search (convex, deterministic, no randomness).

Selection rule (fixed before looking at any result):
  1. On the proxy tuning slice, 5-fold cross-validation (folds from random.Random(42) over the
     sorted query IDs) compares
       (a) ens3: (1 - w) z(stage 1) + w * mean(z(CEs)), w chosen on the training folds
       (b) the learned linear fusion, trained on the training folds.
     (b) is used only if its cross-validated Recall@10 is higher; ties go to (a), the simpler one.
  2. The chosen method is refitted on the whole proxy tuning slice, proxy validation is reported
     once, and the test file is written.
Training data is the proxy tuning slice only (training queries); test inputs are never used to
fit anything.
"""

import argparse
from pathlib import Path
import platform
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from review_retrieval.common import read_json, write_json, write_submission
from review_retrieval.metrics import mean_metrics
from review_retrieval.rerank import WEIGHTS, row_zscore

MODELS = ("rerank", "hardneg", "aug")
FEATURES = ("z(stage 1)", "z(rerank)", "z(hardneg)", "z(aug)", "1/log2(stage-1 rank + 2)")
LAMBDA = 1.0
FOLDS = 5


def features(first, ce):
    """first: (n, K); ce: list of (n, K) -> (n, K, F)."""
    n, k = first.shape
    order = np.argsort(-first, axis=1, kind="stable")
    rank = np.empty_like(order)
    rank[np.arange(n)[:, None], order] = np.arange(k)[None, :]
    cols = [row_zscore(first)] + [row_zscore(s) for s in ce] + [1.0 / np.log2(rank + 2.0)]
    return np.stack(cols, axis=2)


def listwise_loss(x, g, w, lam):
    s = x @ w
    s = s - s.max(axis=1, keepdims=True)
    log_p = s - np.log(np.exp(s).sum(axis=1, keepdims=True))
    return float(-log_p[np.arange(len(g)), g].sum() + 0.5 * lam * w @ w)


def fit_listwise(x, gold_pos, lam=LAMBDA, iterations=100):
    """x: (n, K, F); gold_pos: (n,) position of the correct product or -1 (skipped)."""
    keep = gold_pos >= 0
    x, g = x[keep], gold_pos[keep]
    f = x.shape[2]
    w = np.zeros(f)
    loss = listwise_loss(x, g, w, lam)
    for _ in range(iterations):
        s = x @ w
        s -= s.max(axis=1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(axis=1, keepdims=True)
        mean_x = np.einsum("nk,nkf->nf", p, x)
        grad = (mean_x - x[np.arange(len(g)), g]).sum(axis=0) + lam * w
        hess = np.einsum("nk,nkf,nkg->fg", p, x, x) - mean_x.T @ mean_x + lam * np.eye(f)
        step, t = np.linalg.solve(hess, grad), 1.0
        while t > 1e-10:  # backtracking (Armijo) line search
            new = listwise_loss(x, g, w - t * step, lam)
            if new <= loss - 1e-4 * t * grad @ step:
                break
            t *= 0.5
        w, improvement, loss = w - t * step, loss - new, new
        if improvement < 1e-9 * max(1.0, abs(loss)):
            break
    return w


def ranked(items, scores):
    order = np.argsort(-scores, axis=1, kind="stable")[:, :10]
    return [[row[p] for p in pos] for row, pos in zip(items, order)]


def recall10(items, gold, scores):
    return float(np.mean([g in r for g, r in zip(gold, ranked(items, scores))]))


def ens3_scores(x, w):
    """Same as scripts/run_memory_eval.py: (1 - w) z(stage 1) + w z(mean of the CE z-scores)."""
    return (1.0 - w) * x[:, :, 0] + w * row_zscore(x[:, :, 1:4].mean(axis=2))


def choose_w(items, gold, x):
    curve = {w: recall10(items, gold, ens3_scores(x, w)) for w in WEIGHTS}
    return max(WEIGHTS, key=lambda w: (curve[w], -w)), curve


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "project_release")
    parser.add_argument("--memory-dir", type=Path, default=ROOT / "outputs/memory")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/fusion")
    parser.add_argument("--student-id", default="2024320020")
    args = parser.parse_args()
    if not args.student_id.isalnum():
        parser.error("--student-id must be alphanumeric")
    d = args.memory_dir
    gold_by_id = {q["query_id"]: q["item_id"] for q in read_json(args.data_dir / "train_gold.json")}
    catalog_ids = sorted(p["item_id"] for p in read_json(args.data_dir / "catalog.json"))

    def load(split):
        c = read_json(d / f"candidates_exp_{split}.json")
        x = features(np.asarray(c["first_stage_scores"], dtype=np.float64),
                     [np.load(d / "scores" / f"{m}_exp_{split}.npy") for m in MODELS])
        gold = [gold_by_id.get(q) for q in c["query_ids"]] if split != "test" else None
        pos = None if gold is None else np.array([row.index(g) if g in row else -1 for row, g in zip(c["items"], gold)])
        return c, x, gold, pos

    tune, xt, gt, pt = load("proxy_tune")
    ids = sorted(tune["query_ids"])
    shuffled = ids[:]
    random.Random(42).shuffle(shuffled)
    fold_of = {q: i % FOLDS for i, q in enumerate(shuffled)}
    folds = np.array([fold_of[q] for q in tune["query_ids"]])

    cv = {"ens3": [], "learned": []}
    for k in range(FOLDS):
        tr, te = folds != k, folds == k
        items_tr = [r for r, m in zip(tune["items"], tr) if m]
        items_te = [r for r, m in zip(tune["items"], te) if m]
        gold_tr = [g for g, m in zip(gt, tr) if m]
        gold_te = [g for g, m in zip(gt, te) if m]
        w, _ = choose_w(items_tr, gold_tr, xt[tr])
        cv["ens3"].append(recall10(items_te, gold_te, ens3_scores(xt[te], w)))
        coef = fit_listwise(xt[tr], pt[tr])
        cv["learned"].append(recall10(items_te, gold_te, xt[te] @ coef))
        print(f"fold {k}: ens3 (w {w}) {cv['ens3'][-1]:.4f}  learned {cv['learned'][-1]:.4f}", flush=True)
    mean_cv = {m: float(np.mean(v)) for m, v in cv.items()}
    method = "learned" if mean_cv["learned"] > mean_cv["ens3"] else "ens3"
    print(f"Cross-validated proxy tuning R@10: {mean_cv} -> {method}", flush=True)

    w, curve = choose_w(tune["items"], gt, xt)
    coef = fit_listwise(xt, pt)
    print(f"Refit on the whole proxy tuning slice: ens3 w {w}; learned weights "
          f"{dict(zip(FEATURES, np.round(coef, 4).tolist()))}", flush=True)

    valid, xv, gv, _ = load("proxy_validation")
    flags = valid["has_memory"]
    report = {}
    for name, s in (("ens3", ens3_scores(xv, w)), ("learned", xv @ coef)):
        lists = ranked(valid["items"], s)
        report[name] = {}
        for part, keep in (("all", None), ("with_memory", True), ("without_memory", False)):
            idx = [i for i, f in enumerate(flags) if keep is None or f == keep]
            report[name][part] = {k: round(v, 5) for k, v in mean_metrics([gv[i] for i in idx], [lists[i] for i in idx]).items()}
        print(f"  proxy validation {name}: R@10 {report[name]['all']['recall@10']:.4f} "
              f"(memory {report[name]['with_memory']['recall@10']:.4f}, no memory {report[name]['without_memory']['recall@10']:.4f})",
              flush=True)

    test, xs, _, _ = load("test")
    files = {}
    for name, s in (("ens3", ens3_scores(xs, w)), ("learned", xs @ coef)):
        files[name] = write_submission(args.output_dir / name / f"{args.student_id}_final.csv",
                                       test["query_ids"], ranked(test["items"], s), catalog_ids)
    write_json(args.output_dir / "metrics.json", {
        "method": "listwise linear fusion of stage 1 and three cross-encoders vs ens3, chosen by 5-fold CV on the proxy tuning slice",
        "features": list(FEATURES), "lambda": LAMBDA, "folds": FOLDS,
        "cv_recall_at_10": cv, "cv_mean": mean_cv, "selected": method,
        "ens3_weight": w, "ens3_curve": curve, "learned_weights": dict(zip(FEATURES, coef.tolist())),
        "proxy_validation": report, "test_csv": files,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    })
    print(f"Selected: {method} -> {args.output_dir / method / (args.student_id + '_final.csv')}", flush=True)


if __name__ == "__main__":
    main()
