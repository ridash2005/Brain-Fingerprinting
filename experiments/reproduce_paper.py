# -*- coding: utf-8 -*-
"""Seeded reproduction of the experiments reported in the paper.

Every function that produces a published number is copied from
ConvAE_SDL/ConvAE_SDL.py or ConvAE_SDL/ConvAE_SDL_CrossTask.py.  Three changes
only, none of which alters the arithmetic: batching (to fit in 8 GB of RAM),
fixed seeds, and the K-SVD atom update taking its leading singular vector from
the small Gram matrix instead of a full SVD (same result, ~13x faster).

Stage A   Table 1 / Table 3 protocol.  The ConvAE is trained on the
          resting-state connectomes of all 339 subjects (20 epochs), the K-SVD
          dictionary on the task residuals of all 339 subjects (5 iterations,
          seed 42), and the refined task connectome is matched against the raw
          resting-state connectome.  This is what run_complete_analysis() does.
          Also stores the dictionaries for the atom analysis (Fig. 3).
Stage B   Subject-disjoint 5-fold control of the Stage A protocol ("Generalization
          to Held-Out Subjects").  ConvAE and dictionary are fitted on 4/5 of
          the subjects; the held-out fifth is coded with the frozen dictionary
          and identified against the full 339-subject resting-state catalogue.
Stage D   Table 2 protocol (ConvAE and dictionary fitted on the source task,
          applied to the target task), plus how much of each target's residual
          the source dictionary explains ("Determinants of Cross-Task Transfer").
Stage A2  Stage A with two more seeds, for seed-to-seed spread.

Results are written to experiments/output/results.json after every step, so an
interrupted run resumes where it stopped.  The values from our own run are in
experiments/reference/results.json.

Run from the repository root (FC_DATA/ must hold the connectomes, see README):
    python experiments/reproduce_paper.py A B D A2
"""
import gc
import io
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import orthogonal_mp
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FC_DIR = os.path.join(ROOT, "FC_DATA")
OUTDIR = os.path.join(HERE, "output")
ART = os.path.join(OUTDIR, "artifacts")
OUT = os.path.join(OUTDIR, "results.json")
LOG = os.path.join(OUTDIR, "reproduce_paper.log")
os.makedirs(ART, exist_ok=True)

DOMAINS = ["wm", "emotion", "gambling", "relational"]
TUNED = {"wm": (15, 13), "emotion": (15, 13), "gambling": (14, 11), "relational": (11, 9)}
P = 360
TRIL = np.tril_indices(P, k=-1)
TRIU = np.triu_indices(P, k=1)
torch.set_num_threads(os.cpu_count() or 4)


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with io.open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_results():
    if os.path.exists(OUT):
        with io.open(OUT, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(res):
    tmp = OUT + ".tmp"
    with io.open(tmp, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, sort_keys=True)
    os.replace(tmp, OUT)


def load_fc(name):
    return np.load(os.path.join(FC_DIR, "fc_%s.npy" % name))


# ---------------------------------------------------------------- model ----
class ConvAutoencoder(nn.Module):
    """Identical to ConvAE_SDL.py (33,969 parameters)."""

    def __init__(self, n_parcels=360):
        super().__init__()
        self.n = n_parcels
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2), nn.Tanh())

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_convae(fc, seed, epochs=20):
    """Adam 1e-3, batch 16, MSE, shuffled -- as in ConvAE_SDL.py."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ConvAutoencoder(P)
    x = torch.tensor(fc[:, np.newaxis, :, :], dtype=torch.float32)
    loader = DataLoader(TensorDataset(x, x), batch_size=16, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=0.001)
    crit = nn.MSELoss()
    model.train()
    t0 = time.time()
    for ep in range(epochs):
        tot = 0.0
        for batch, _ in loader:
            opt.zero_grad()
            loss = crit(model(batch), batch)
            loss.backward()
            opt.step()
            tot += loss.item() * batch.shape[0]
        if ep in (0, epochs - 1):
            log("    epoch %2d/%d  mse=%.5f  (%.0fs)" % (ep + 1, epochs, tot / len(x), time.time() - t0))
    model.eval()
    del x, loader
    return model


def reconstruct(model, fc, bs=16):
    out = np.empty(fc.shape, dtype=np.float32)
    with torch.no_grad():
        for s in range(0, fc.shape[0], bs):
            b = torch.tensor(fc[s:s + bs, np.newaxis], dtype=torch.float32)
            out[s:s + bs] = model(b).numpy()[:, 0]
    return out


# --------------------------------------------------------------- K-SVD ----
def omp_sparse_coding(Y, D, L):
    # orthogonal_mp with a 2-D target loops over the columns with the same
    # Cholesky OMP as the per-column call in ConvAE_SDL.py
    return np.asarray(orthogonal_mp(D, Y, n_nonzero_coefs=L)).reshape(D.shape[1], Y.shape[1])


def top_singular_triplet(E):
    """Leading (u, s, v) of E via the small Gram matrix E^T E.

    Gives the same rank-one term u s v^T as np.linalg.svd(E)[:, 0] (checked to
    1e-13 on residual-sized matrices) but is ~13x faster for 64,620 x n blocks,
    because only the n x n Gram matrix is decomposed.
    """
    w, V = np.linalg.eigh(E.T @ E)
    s = np.sqrt(max(w[-1], 0.0))
    if s < 1e-12:
        U, S, Vt = np.linalg.svd(E, full_matrices=False)
        return U[:, 0], S[0], Vt[0, :]
    v = V[:, -1]
    return E @ v / s, s, v


def update_dictionary(Y, D, X):
    for k in range(D.shape[1]):
        idx = np.nonzero(X[k, :])[0]
        if len(idx) == 0:
            D[:, k] = np.random.randn(D.shape[0])
            D[:, k] /= np.linalg.norm(D[:, k])
            continue
        residual = Y[:, idx] - np.dot(D, X[:, idx])
        residual += np.outer(D[:, k], X[k, idx])
        u, s, v = top_singular_triplet(residual)
        D[:, k] = u
        X[k, idx] = s * v
    return D, X


def k_svd(Y, K, L, n_iter, random_state=42):
    np.random.seed(random_state)
    m, n = Y.shape
    D = np.random.randn(m, K)
    D = D / np.linalg.norm(D, axis=0, keepdims=True)
    X = None
    for _ in range(n_iter):
        X = omp_sparse_coding(Y, D, L)
        D, X = update_dictionary(Y, D, X)
    return D, X


def sym(tri_elements):
    """reconstruct_symmetric_matrix() from ConvAE_SDL.py (diagonal set to 1)."""
    m = np.zeros((P, P))
    m[TRIL] = tri_elements
    m += m.T
    np.fill_diagonal(m, 1.0)
    return m


def tri_matrix(mats):
    return np.stack([m[TRIL] for m in mats], axis=1).astype(np.float64)


# -------------------------------------------------------- identification ----
def zrows(mats):
    """Row-standardised flattened matrices; zrows(a) @ zrows(b).T == np.corrcoef."""
    F = np.array(mats, dtype=np.float64).reshape(len(mats), -1)
    F -= F.mean(axis=1, keepdims=True)
    F /= np.linalg.norm(F, axis=1, keepdims=True)
    return F


def similarity(probe, gallery_z):
    c = zrows(probe) @ gallery_z.T
    return np.nan_to_num(c, nan=-1.0)


def correct_vector(sim, true_idx):
    return (np.argmax(sim, axis=1) == np.asarray(true_idx)).astype(int)


def pct(x):
    return round(100.0 * float(np.mean(x)), 2)


def id_stats(sim):
    """Identifiability summary of a square probe x gallery similarity matrix."""
    n = sim.shape[0]
    off = ~np.eye(n, dtype=bool)
    z = (sim - sim.mean(1, keepdims=True)) / sim.std(1, keepdims=True)
    return {"I_self": round(float(np.mean(np.diag(sim))), 4),
            "I_others": round(float(np.mean(sim[off])), 4),
            "I_diff": round(float(np.mean(np.diag(sim)) - np.mean(sim[off])), 4),
            "median_diag_z": round(float(np.median(np.diag(z))), 3)}


def finn_bidirectional(fc_task, fc_rest):
    a = np.array([m[TRIU] for m in fc_task])
    b = np.array([m[TRIU] for m in fc_rest])
    n = len(a)
    c = np.corrcoef(a, b)[:n, n:]
    t2r = np.mean(np.argmax(c, axis=1) == np.arange(n))
    r2t = np.mean(np.argmax(c, axis=0) == np.arange(n))
    return round(100 * (t2r + r2t) / 2, 2), round(100 * t2r, 2)


# --------------------------------------------------------------- stages ----
def stage_A(res, seed=0, tag="A"):
    """Table 1 protocol on the full cohort (transductive)."""
    fc_rest = load_fc("rest")
    G = zrows(fc_rest)
    n = fc_rest.shape[0]
    log("== Stage %s: ConvAE on resting-state connectomes of all %d subjects (seed %d)" % (tag, n, seed))
    model = train_convae(fc_rest, seed=seed)
    if tag == "A":
        torch.save(model.state_dict(), os.path.join(ART, "convae_rest_full_seed%d.pt" % seed))
    out = res.setdefault(tag, {}).setdefault("seed%d" % seed, {})
    for task in DOMAINS:
        if task in out and "convae_sdl" in out[task]:
            log("  %s done" % task)
            continue
        K, L = TUNED[task]
        fc = load_fc(task)
        cell = {"K": K, "L": L}
        t0 = time.time()
        if tag == "A":
            cell["finn_bidirectional"], cell["raw_task_to_rest"] = finn_bidirectional(fc, fc_rest)
            # SDL on raw FC (the ablation's "SDL only"): both D.x and FC - D.x
            Yr = tri_matrix(fc)
            Dr, Xr = k_svd(Yr, K, L, n_iter=5)
            rec = (Dr @ Xr).T
            sdl_rec = np.stack([sym(rec[i]) for i in range(n)])
            cell["sdl_only_reconstruction"] = pct(correct_vector(similarity(sdl_rec, G), np.arange(n)))
            del sdl_rec
            sdl_ref = np.stack([fc[i] - sym(rec[i]) for i in range(n)])
            cell["sdl_only_refinement"] = pct(correct_vector(similarity(sdl_ref, G), np.arange(n)))
            del sdl_ref, Yr, Dr, Xr, rec
            gc.collect()
        recon = reconstruct(model, fc)
        resid = fc - recon
        sim_raw = similarity(fc, G)
        sim_res = similarity(resid, G)
        cell["raw_one_direction"] = pct(correct_vector(sim_raw, np.arange(n)))
        cell["convae_residuals"] = pct(correct_vector(sim_res, np.arange(n)))
        Y = tri_matrix(resid)
        D, X = k_svd(Y, K, L, n_iter=5)
        DX = D @ X
        refined = np.stack([resid[i] - sym(DX[:, i]) for i in range(n)])
        sim_ref = similarity(refined, G)
        cell["convae_sdl"] = pct(correct_vector(sim_ref, np.arange(n)))
        dx_only = np.stack([sym(DX[:, i]) for i in range(n)])
        cell["residual_dictionary_part_only"] = pct(correct_vector(similarity(dx_only, G), np.arange(n)))
        del dx_only
        cell["dictionary_explained_energy"] = round(float(1 - np.sum((Y - DX) ** 2) / np.sum(Y ** 2)), 4)
        cell["code_nonzeros_mean"] = round(float(np.mean(np.sum(X != 0, axis=0))), 2)
        if tag == "A":
            cell["id_raw"] = id_stats(sim_raw)
            cell["id_residual"] = id_stats(sim_res)
            cell["id_refined"] = id_stats(sim_ref)
            # inter-subject similarity of the task-side representations themselves
            for nm, mats in (("raw", fc), ("residual", resid), ("refined", refined)):
                Z = zrows(mats)
                S = Z @ Z.T
                cell["mean_between_subject_r_" + nm] = round(float(S[~np.eye(n, dtype=bool)].mean()), 4)
                del Z, S
            # top-K principal subspace of the residuals vs the dictionary span
            Yc = Y  # K-SVD sees uncentred data, so compare with the uncentred SVD
            U, s, _ = np.linalg.svd(Yc, full_matrices=False)
            Uk = U[:, :K]
            Q, _ = np.linalg.qr(D)
            cell["dict_vs_topK_svd_overlap"] = round(float(np.sum((Uk.T @ Q) ** 2) / K), 4)
            cell["topK_svd_energy"] = round(float(np.sum(s[:K] ** 2) / np.sum(s ** 2)), 4)
            del U, s, Uk, Q
            np.savez_compressed(
                os.path.join(ART, "stageA_%s.npz" % task),
                D=D.astype(np.float32), X=X.astype(np.float32),
                mean_fc=fc.mean(0).astype(np.float32),
                mean_resid=resid.mean(0).astype(np.float32),
                mean_recon=recon.mean(0).astype(np.float32),
                ex_fc=fc[0].astype(np.float32), ex_recon=recon[0].astype(np.float32),
                ex_resid=resid[0].astype(np.float32), ex_dx=sym(DX[:, 0]).astype(np.float32),
                ex_refined=refined[0].astype(np.float32),
                sim_raw=sim_raw.astype(np.float32), sim_res=sim_res.astype(np.float32),
                sim_ref=sim_ref.astype(np.float32),
                correct_refined=correct_vector(sim_ref, np.arange(n)))
        out[task] = cell
        save_results(res)
        log("  %-10s raw=%.2f  resid=%.2f  ConvAE+SDL=%.2f  (%.0fs)  %s" % (
            task, cell["raw_one_direction"], cell["convae_residuals"], cell["convae_sdl"], time.time() - t0,
            "" if tag != "A" else "SDL[Dx]=%.2f SDL[FC-Dx]=%.2f finn=%.2f" % (
                cell["sdl_only_reconstruction"], cell["sdl_only_refinement"], cell["finn_bidirectional"])))
        del fc, recon, resid, sim_raw, sim_res, sim_ref, Y, D, X, DX, refined
        gc.collect()
    del model, G, fc_rest
    gc.collect()


def stage_B(res, n_folds=5, seed=0):
    """Subject-disjoint control: nothing fitted on the held-out subjects."""
    fc_rest = load_fc("rest")
    G = zrows(fc_rest)
    n = fc_rest.shape[0]
    perm = np.random.RandomState(seed).permutation(n)
    folds = np.array_split(perm, n_folds)
    out = res.setdefault("B", {"folds": [f.tolist() for f in folds]})
    log("== Stage B: subject-disjoint %d-fold control (catalogue = all %d resting-state scans)" % (n_folds, n))
    for f, test_idx in enumerate(folds):
        key = "fold%d" % f
        if key in out and all(t in out[key] for t in DOMAINS):
            log("  %s done" % key)
            continue
        train_idx = np.setdiff1d(perm, test_idx)
        log("  fold %d/%d  train=%d  held-out=%d" % (f + 1, n_folds, len(train_idx), len(test_idx)))
        model = train_convae(fc_rest[train_idx], seed=seed + f)
        fo = out.setdefault(key, {})
        for task in DOMAINS:
            if task in fo:
                continue
            K, L = TUNED[task]
            fc = load_fc(task)
            resid = fc - reconstruct(model, fc)
            Ytr = tri_matrix(resid[train_idx])
            D, _ = k_svd(Ytr, K, L, n_iter=5)
            Yte = tri_matrix(resid[test_idx])
            Xte = omp_sparse_coding(Yte, D, L)
            DX = D @ Xte
            refined = np.stack([resid[i] - sym(DX[:, j]) for j, i in enumerate(test_idx)])
            fo[task] = {
                "raw": correct_vector(similarity(fc[test_idx], G), test_idx).tolist(),
                "residual": correct_vector(similarity(resid[test_idx], G), test_idx).tolist(),
                "convae_sdl": correct_vector(similarity(refined, G), test_idx).tolist()}
            save_results(res)
            log("    %-10s raw=%.2f  resid=%.2f  ConvAE+SDL=%.2f" % (
                task, pct(fo[task]["raw"]), pct(fo[task]["residual"]), pct(fo[task]["convae_sdl"])))
            del fc, resid, Ytr, D, Yte, Xte, DX, refined
            gc.collect()
        del model
        gc.collect()
    summ = {}
    for task in DOMAINS:
        summ[task] = {m: pct(np.concatenate([out["fold%d" % f][task][m] for f in range(n_folds)]))
                      for m in ("raw", "residual", "convae_sdl")}
    summ["average"] = {m: round(float(np.mean([summ[t][m] for t in DOMAINS])), 2)
                       for m in ("raw", "residual", "convae_sdl")}
    out["summary"] = summ
    save_results(res)
    log("  Stage B summary: %s" % json.dumps(summ))
    del G, fc_rest
    gc.collect()


def stage_D(res, seed=0):
    """Table 2 protocol: fit on the source task, apply to the target task."""
    fc_rest = load_fc("rest")
    G = zrows(fc_rest)
    n = fc_rest.shape[0]
    out = res.setdefault("D", {})
    log("== Stage D: cross-task protocol (ConvAE + dictionary fitted on the source task)")
    for src in DOMAINS:
        if src in out and all(t in out[src] for t in DOMAINS if t != src):
            log("  source %s done" % src)
            continue
        K, L = TUNED[src]
        fc_s = load_fc(src)
        model = train_convae(fc_s, seed=seed)
        resid_s = fc_s - reconstruct(model, fc_s)
        Ys = tri_matrix(resid_s)
        D, Xs = k_svd(Ys, K, L, n_iter=10)
        so = out.setdefault(src, {})
        so["self_explained_energy"] = round(float(1 - np.sum((Ys - D @ Xs) ** 2) / np.sum(Ys ** 2)), 4)
        Q, _ = np.linalg.qr(D)
        np.save(os.path.join(ART, "stageD_dict_%s.npy" % src), D.astype(np.float32))
        del fc_s, resid_s, Ys, Xs
        gc.collect()
        for tgt in DOMAINS:
            if tgt == src or tgt in so:
                continue
            fc_t = load_fc(tgt)
            resid_t = fc_t - reconstruct(model, fc_t)
            Yt = tri_matrix(resid_t)
            Xt = omp_sparse_coding(Yt, D, L)
            DX = D @ Xt
            refined = np.stack([resid_t[i] - sym(DX[:, i]) for i in range(n)])
            mres = Yt.mean(axis=1)
            so[tgt] = {
                "convae_sdl": pct(correct_vector(similarity(refined, G), np.arange(n))),
                "explained_energy": round(float(1 - np.sum((Yt - DX) ** 2) / np.sum(Yt ** 2)), 4),
                "mean_residual_in_dict_span": round(float(np.linalg.norm(Q.T @ mres) / np.linalg.norm(mres)), 4)}
            save_results(res)
            log("    %s -> %-10s acc=%.2f  explained=%.3f  mean-resid-in-span=%.3f" % (
                src, tgt, so[tgt]["convae_sdl"], so[tgt]["explained_energy"], so[tgt]["mean_residual_in_dict_span"]))
            del fc_t, resid_t, Yt, Xt, DX, refined
            gc.collect()
        del model, D, Q
        gc.collect()
    del G, fc_rest
    gc.collect()


def main(stages):
    res = load_results()
    for st in stages:
        t0 = time.time()
        if st == "A":
            stage_A(res, seed=0, tag="A")
        elif st == "B":
            stage_B(res)
        elif st == "D":
            stage_D(res)
        elif st == "A2":
            for sd in (1, 2):
                stage_A(res, seed=sd, tag="A2")
        log("stage %s finished in %.1f min" % (st, (time.time() - t0) / 60))


if __name__ == "__main__":
    main(sys.argv[1:] or ["A", "B", "D", "A2"])
