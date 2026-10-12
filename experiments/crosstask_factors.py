# -*- coding: utf-8 -*-
"""What explains the spread of the cross-task accuracies (Table 2)?

Tests candidate factors against the 12 off-diagonal entries of Table 2 of the
paper (the published values, hard-coded below):

  target-side   scan length of the target task; raw identifiability of the
                target task (raw task connectome vs. raw resting-state catalog)
  pair-side     similarity of the group-average connectomes of the two tasks;
                within-subject similarity of a person's connectomes in the two
                tasks; how much of the target's residual the source dictionary
                explains (from Stage D of reproduce_paper.py, if present)

Memory-light: connectomes are read through np.load(mmap_mode="r").

    python experiments/crosstask_factors.py
"""
import io
import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FC_DIR = os.path.join(ROOT, "FC_DATA")
RESULTS = os.path.join(HERE, "output", "results.json")
OUT = os.path.join(HERE, "output", "crosstask_factors.json")

TASKS = ["wm", "emotion", "gambling", "relational"]
FRAMES = {"wm": 2 * 405, "emotion": 2 * 176, "gambling": 2 * 253, "relational": 2 * 232}
DAY = {"wm": 1, "gambling": 1, "emotion": 2, "relational": 2}
# Table 2 of the paper: rows = training task, columns = evaluation target
TABLE2 = {
    "wm": {"emotion": 61.06, "gambling": 73.75, "relational": 65.49},
    "emotion": {"wm": 69.62, "gambling": 72.27, "relational": 61.06},
    "gambling": {"wm": 71.09, "emotion": 70.80, "relational": 66.96},
    "relational": {"wm": 73.45, "emotion": 65.78, "gambling": 78.76},
}
TABLE1 = {"wm": 76.99, "emotion": 74.04, "gambling": 75.22, "relational": 68.44}
FINN = {"wm": 41.74, "emotion": 28.17, "gambling": 32.30, "relational": 23.01}
TRIL = np.tril_indices(360, k=-1)


def mm(name):
    return np.load(os.path.join(FC_DIR, "fc_%s.npy" % name), mmap_mode="r")


def edges(m):
    return np.asarray(m[TRIL], dtype=np.float64)


def zs(v):
    v = v - v.mean()
    return v / np.linalg.norm(v)


def main():
    res = json.load(io.open(RESULTS)) if os.path.exists(RESULTS) else {}
    arr = {t: mm(t) for t in TASKS + ["rest"]}
    n = arr["rest"].shape[0]

    # group-average connectomes
    gmean = {}
    for t, a in arr.items():
        acc = np.zeros(len(TRIL[0]))
        for i in range(n):
            acc += edges(a[i])
        gmean[t] = acc / n

    # within-subject similarity between conditions, and group-mean similarity
    within, between_means = {}, {}
    conds = TASKS + ["rest"]
    for a_i, a in enumerate(conds):
        for b in conds[a_i + 1:]:
            r = [float(zs(edges(arr[a][i])) @ zs(edges(arr[b][i]))) for i in range(n)]
            key = "%s|%s" % (a, b)
            within[key] = float(np.mean(r))
            between_means[key] = float(zs(gmean[a]) @ zs(gmean[b]))

    def pair(d, a, b):
        return d.get("%s|%s" % (a, b), d.get("%s|%s" % (b, a)))

    rows = []
    D = res.get("D", {})
    for s in TASKS:
        for t in TASKS:
            if s == t:
                continue
            rows.append({
                "source": s, "target": t, "acc": TABLE2[s][t],
                "target_frames": FRAMES[t], "target_raw_finn": FINN[t], "target_within_table1": TABLE1[t],
                "source_frames": FRAMES[s], "same_day": int(DAY[s] == DAY[t]),
                "groupmean_r": pair(between_means, s, t), "within_subject_r": pair(within, s, t),
                "target_rest_within_subject_r": pair(within, t, "rest"),
                "rerun_acc": D.get(s, {}).get(t, {}).get("convae_sdl"),
                "explained_energy": D.get(s, {}).get(t, {}).get("explained_energy"),
                "mean_resid_in_span": D.get(s, {}).get(t, {}).get("mean_residual_in_dict_span"),
            })
    acc = np.array([r["acc"] for r in rows])

    # additive decomposition: acc(s -> t) ~ mu + row_s + col_t
    X = []
    for r in rows:
        x = [1.0] + [float(r["source"] == s) for s in TASKS[1:]] + [float(r["target"] == t) for t in TASKS[1:]]
        X.append(x)
    X = np.array(X)
    beta, *_ = np.linalg.lstsq(X, acc, rcond=None)
    fit = X @ beta
    r2_add = 1 - np.sum((acc - fit) ** 2) / np.sum((acc - acc.mean()) ** 2)
    Xc = X[:, [0] + list(range(4, 7))]
    bc, *_ = np.linalg.lstsq(Xc, acc, rcond=None)
    r2_col = 1 - np.sum((acc - Xc @ bc) ** 2) / np.sum((acc - acc.mean()) ** 2)
    Xr = X[:, :4]
    br, *_ = np.linalg.lstsq(Xr, acc, rcond=None)
    r2_row = 1 - np.sum((acc - Xr @ br) ** 2) / np.sum((acc - acc.mean()) ** 2)

    col_mean = {t: float(np.mean([r["acc"] for r in rows if r["target"] == t])) for t in TASKS}
    row_mean = {s: float(np.mean([r["acc"] for r in rows if r["source"] == s])) for s in TASKS}

    def corr(key, sub=None):
        rr = [r for r in rows if r.get(key) is not None]
        if len(rr) < 4:
            return None
        x = np.array([r[key] for r in rr], dtype=float)
        y = np.array([r["acc"] for r in rr])
        pr = stats.pearsonr(x, y)
        sr = stats.spearmanr(x, y)
        return {"n": len(rr), "pearson_r": round(float(pr[0]), 3), "pearson_p": round(float(pr[1]), 4),
                "spearman_rho": round(float(sr[0]), 3), "spearman_p": round(float(sr[1]), 4)}

    out = {
        "rows": rows,
        "groupmean_similarity": between_means,
        "within_subject_similarity": within,
        "column_means": col_mean, "row_means": row_mean,
        "r2_additive_row_plus_col": round(float(r2_add), 3),
        "r2_columns_only": round(float(r2_col), 3), "r2_rows_only": round(float(r2_row), 3),
        "binomial_se_pct_at_70": round(float(100 * np.sqrt(0.7 * 0.3 / n)), 2),
        "correlations": {k: corr(k) for k in ("target_frames", "target_raw_finn", "target_within_table1",
                                             "source_frames", "same_day", "groupmean_r", "within_subject_r",
                                             "target_rest_within_subject_r", "explained_energy",
                                             "mean_resid_in_span", "rerun_acc")},
    }
    # residual (pair-specific) part after removing the target column effect
    resid = acc - np.array([col_mean[r["target"]] for r in rows])
    for key in ("groupmean_r", "within_subject_r", "explained_energy", "mean_resid_in_span", "source_frames"):
        rr = [(r[key], e) for r, e in zip(rows, resid) if r.get(key) is not None]
        if len(rr) >= 4:
            x, y = np.array(rr).T
            out["correlations"]["after_target_effect__" + key] = {
                "pearson_r": round(float(stats.pearsonr(x, y)[0]), 3),
                "pearson_p": round(float(stats.pearsonr(x, y)[1]), 4)}
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
