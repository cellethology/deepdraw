"""Binary top-decile SVM and entropy selection."""

import numpy as np
from scipy.special import xlogy
from sklearn.svm import SVC


def binary_labels(scores, ids):
    scores, ids = np.asarray(scores, float), np.asarray(ids)
    if len(scores) < 10 or not np.all(np.isfinite(scores)):
        raise ValueError("At least ten finite phenotypes required.")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Circuit IDs must be unique; equal sequences remain distinct.")
    order = np.lexsort((ids, scores))
    labels = np.empty(len(scores), dtype=np.int32)
    labels[order] = np.arange(len(scores)) * 10 // len(scores) == 9
    return labels


def top_targets(scores):
    scores = np.asarray(scores, float)
    count = max(1, int(len(scores) * 0.01))
    cutoff = float(np.sort(scores)[-count])
    return scores >= cutoff, cutoff


def entropy(probabilities):
    p = np.asarray(probabilities, float)
    if p.ndim != 2 or not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("Invalid class probabilities.")
    if not np.allclose(p.sum(axis=1), 1, atol=1e-8):
        raise ValueError("Class probabilities do not sum to one.")
    return -np.sum(xlogy(p, p), axis=1) / np.log(2)


def select_entropy(pool, probabilities, rng, batch_size=12):
    h = entropy(probabilities)
    order = np.lexsort((rng.random(len(pool)), -h))
    chosen = order[: min(batch_size, len(pool))]
    return np.asarray(pool)[chosen], h[chosen]


def run(
    kernel,
    ids,
    scores,
    labels,
    seed,
    total_rounds=200,
    *,
    stop_on_hit=True,
    progress=None,
):
    n = len(ids)
    if n < 12 or total_rounds < 1:
        raise ValueError(
            "At least twelve candidates and a positive batch budget required"
        )
    target, _ = top_targets(scores)
    selection_rng = np.random.default_rng(seed)
    tie_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    calibration_rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    batch = selection_rng.choice(n, size=12, replace=False)
    measured = np.zeros(n, dtype=bool)
    train = []
    columns = np.empty((n, min(n, total_rounds * 12)), dtype=np.float64)
    computed = 0
    first_hit = None
    history = []
    acquisition_entropies = None
    fitted_classes = []
    for round_index in range(total_rounds):
        if np.any(measured[batch]) or len(np.unique(batch)) != len(batch):
            raise RuntimeError("Acquisition repeated a measured circuit.")
        measured[batch] = True
        train.extend(batch.tolist())
        hits = ids[batch[target[batch]]].tolist()
        if hits and first_hit is None:
            first_hit = round_index + 1
        entry = {
            "total_rounds": round_index + 1,
            "circuit_ids": ids[batch].tolist(),
            "phenotypes": scores[batch].tolist(),
            "class_labels": labels[batch].tolist(),
            "fitted_classes": fitted_classes,
            "acquisition_entropies": acquisition_entropies,
        }
        history.append(entry)
        if progress:
            progress(history)
        if (stop_on_hit and first_hit is not None) or round_index + 1 == total_rounds:
            break
        pool = np.flatnonzero(~measured)
        if not len(pool):
            break
        y = labels[train]
        fitted_classes = np.unique(y).tolist()
        if len(fitted_classes) == 1:
            probabilities = np.ones((len(pool), 1))
        else:
            added = np.asarray(train[computed:], dtype=np.int32)
            columns[:, computed : len(train)] = kernel.gram(np.arange(n), added)
            computed = len(train)
            svm = SVC(
                kernel="precomputed",
                probability=True,
                C=1.0,
                random_state=int(calibration_rng.integers(0, 2**31 - 1)),
            )
            svm.fit(np.ascontiguousarray(columns[train, : len(train)]), y)
            if not np.array_equal(svm.classes_, np.unique(y)):
                raise RuntimeError("Unexpected SVM classes.")
            probabilities = svm.predict_proba(
                np.ascontiguousarray(columns[pool, : len(train)])
            )
        batch, h = select_entropy(pool, probabilities, tie_rng)
        acquisition_entropies = h.tolist()
    return {"rounds_to_top": first_hit, "history": history}
