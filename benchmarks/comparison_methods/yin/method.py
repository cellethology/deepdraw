import numpy as np


def log_score(expression, task):
    y = np.asarray(expression, dtype=np.float64)
    if task == "IS":
        return y[:, 1] - y[:, 0]
    if task == "OR":
        return y[:, 1:].min(axis=1) - y[:, 0]
    if task == "AND":
        return y[:, 3] - y[:, :3].max(axis=1)
    raise ValueError("Unknown design task")


def run(
    codes,
    expression,
    scores,
    task,
    seed=0,
    total_rounds=200,
    *,
    settings=None,
    progress=None,
    predictor=None,
):
    codes = np.asarray(codes)
    expression = np.asarray(expression, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(codes)
    outputs = 2 if task == "IS" else 4
    if task not in ("IS", "OR", "AND"):
        raise ValueError("Unknown design task")
    if n < 12 or total_rounds < 1:
        raise ValueError(
            "At least twelve candidates and a positive batch budget required"
        )
    if (
        codes.ndim != 2
        or codes.shape[1] == 0
        or not np.issubdtype(codes.dtype, np.integer)
        or np.any((codes < 0) | (codes > 4))
    ):
        raise ValueError(
            "DNA codes must be a candidate-by-position array with values 0-4"
        )
    if (
        expression.shape != (n, outputs)
        or scores.shape != (n,)
        or not np.isfinite(expression).all()
        or not np.isfinite(scores).all()
    ):
        raise ValueError("Invalid expression or objective measurements")
    count = max(1, int(n * 0.01))
    target = scores >= np.sort(scores)[-count]
    batch = np.random.default_rng(seed).choice(n, 12, replace=False)
    measured = np.zeros(n, dtype=bool)
    observed, history = [], []
    first_hit = None
    for number in range(1, total_rounds + 1):
        measured[batch] = True
        observed.extend(batch.tolist())
        if target[batch].any() and first_hit is None:
            first_hit = number
        history.append({"round": number, "indices": batch.tolist()})
        if progress:
            progress(history)
        remaining = np.flatnonzero(~measured)
        if first_hit is not None or number == total_rounds or not len(remaining):
            break
        if predictor is None:
            from .model import fit_predict

            predictor = fit_predict
        indices = np.asarray(observed)
        predicted = np.asarray(
            predictor(
                codes,
                indices,
                expression[indices].copy(),
                remaining,
                seed,
                number + 1,
                **(settings or {}),
            )
        )
        if (
            predicted.shape != (len(remaining), outputs)
            or not np.isfinite(predicted).all()
        ):
            raise RuntimeError("Invalid ensemble expression predictions")
        predicted_scores = log_score(predicted, task)
        rng = np.random.default_rng(np.random.SeedSequence([seed, number + 1, 4001]))
        order = rng.permutation(len(remaining))
        order = order[np.argsort(-predicted_scores[order], kind="stable")]
        batch = remaining[order[:12]]
    return {"rounds_to_top": first_hit, "history": history}
