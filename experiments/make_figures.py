# -*- coding: utf-8 -*-
"""Figures 1-3 of the paper, drawn from the real data.

Fig. 1  the fingerprinting process (working-memory task, N = 339)
Fig. 2  the ConvAE + SDL architecture with the matrices of one subject
Fig. 3  the three strongest dictionary atoms of the working-memory task

Every matrix and number shown is computed from FC_DATA or from the Stage A
artifacts of reproduce_paper.py, so run that first (stage A).

Each figure is drawn on a canvas of exactly the LNCS text width (122 mm), so
LaTeX includes it at scale 1.0; text is 7-8 pt, so no glyph is below 6 pt after
placement; and every encoding survives black-and-white printing (frame weight
instead of frame colour, a monotonic-lightness colour map where the sign of a
value matters).

    python experiments/make_figures.py fig1 fig2 fig3

Outputs (PDF and PNG) go to experiments/output/figures/.
"""
import io
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FC_DIR = os.path.join(ROOT, "FC_DATA")
ART = os.path.join(HERE, "output", "artifacts")
PAPER = os.path.join(HERE, "output", "figures")
REGIONS = os.path.join(ROOT, "DATA", "hcp_rest", "regions.npy")
RESULTS = os.path.join(HERE, "output", "results.json")

TEXTWIDTH_IN = 122 / 25.4          # LNCS text width
P = 360
TRIL = np.tril_indices(P, k=-1)

# Ji et al. (2019) network order and the labels used in the paper
NET_ORDER = ["Visual1", "Visual2", "Somatomotor", "Cingulo-Oper", "Dorsal-atten", "Language",
             "Frontopariet", "Auditory", "Default", "Posterior-Mu", "Ventral-Mult", "Orbito-Affec"]
NET_SHORT = ["VIS1", "VIS2", "SMN", "CON", "DAN", "LAN", "FPN", "AUD", "DMN", "PMM", "VMM", "ORA"]

INK = "#1d2733"
MUTED = "#6b7785"
BLUE = "#1f5fa8"
ORANGE = "#c8662b"
PURPLE = "#6a4c93"
LIGHT = "#eef2f7"
FS = 7.5          # base lettering (pt)
FS_SMALL = 7.0    # smallest regular lettering (pt)


def style():
    import matplotlib._mathtext as mathtext
    mathtext.SHRINK_FACTOR = 0.8      # first-level scripts of a 7.5-pt label stay at 6 pt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "axes.formatter.use_mathtext": True,
        "font.size": FS,
        "axes.labelsize": FS,
        "axes.titlesize": FS,
        "xtick.labelsize": FS_SMALL,
        "ytick.labelsize": FS_SMALL,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "pdf.fonttype": 42,
        "savefig.dpi": 1200,
    })


def save(fig, name):
    """Fixed canvas (no tight bbox), so the PDF is exactly the LNCS text width."""
    os.makedirs(PAPER, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(PAPER, "%s.%s" % (name, ext)), dpi=1200 if ext == "pdf" else 300)
    plt.close(fig)
    print("wrote", name)


def fc(name, idx):
    return np.asarray(np.load(os.path.join(FC_DIR, "fc_%s.npy" % name), mmap_mode="r")[idx], dtype=np.float64)


FC_NORM = matplotlib.colors.TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=1.0)   # raw FC is mostly positive


def thumb(fig, rect, mat, vmax, cmap="RdBu_r", frame=INK, lw=0.6):
    """vmax=None -> connectome scale (FC_NORM); otherwise symmetric +/- vmax."""
    ax = fig.add_axes(rect)
    m = np.array(mat, dtype=np.float64)
    np.fill_diagonal(m, np.nan)
    if vmax is None:
        ax.imshow(m, cmap=cmap, norm=FC_NORM, interpolation="nearest")
    else:
        ax.imshow(m, cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(frame)
        s.set_linewidth(lw)
    return ax


def arrow(fig, p0, p1, color=INK, lw=0.8, ms=6):
    fig.patches.append(FancyArrowPatch(p0, p1, transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=ms, color=color, lw=lw))


def box(fig, rect, fc_, ec, lw=0.6, r=0.012):
    fig.patches.append(FancyBboxPatch((rect[0], rect[1]), rect[2], rect[3],
                                      boxstyle="round,pad=0,rounding_size=%s" % r,
                                      transform=fig.transFigure, fc=fc_, ec=ec, lw=lw))


# --------------------------------------------------------------- Fig. 1 ----
def fig1():
    """Identification protocol with real data, plus the cohort-level effect."""
    rest = np.load(os.path.join(FC_DIR, "fc_rest.npy"), mmap_mode="r")
    n = rest.shape[0]
    art = np.load(os.path.join(ART, "stageA_wm.npz"))
    sim_ref = art["sim_ref"].astype(np.float64)
    sim_raw = art["sim_raw"].astype(np.float64)

    def self_z(sim):
        z = (sim - sim.mean(1, keepdims=True)) / sim.std(1, keepdims=True)
        return np.diag(z)

    z_raw, z_ref = np.median(self_z(sim_raw)), np.median(self_z(sim_ref))
    ok = art["correct_refined"].astype(bool) & (np.argmax(sim_raw, 1) == np.arange(n))
    cand = np.flatnonzero(ok)
    cand = cand[(cand > 0.1 * n) & (cand < 0.9 * n)]     # keep the highlighted stem away from the axis ends
    margin = np.array([sim_raw[c, c] - np.max(np.delete(sim_raw[c], c)) for c in cand])
    i = int(cand[np.argmin(np.abs(margin - np.median(margin)))])   # representative, not the best case
    probe = fc("wm", i)

    W, H = TEXTWIDTH_IN, 2.05
    fig = plt.figure(figsize=(W, H))
    asp = W / H
    tw = 0.125
    th = tw * asp
    ymid = 0.47
    y_title, y_label = 0.79, 0.165

    # probe
    px = 0.012
    thumb(fig, [px, ymid - th / 2, tw, th], probe, None, lw=0.6)
    fig.text(px + tw / 2, y_title, "Probe: task\nscan of $i$", ha="center", va="bottom",
             fontsize=FS, linespacing=1.1)
    fig.text(px + tw / 2, y_label, r"$C_i^{\mathrm{task}}$", ha="center", va="top", fontsize=8)

    # catalogue: a cascade of real resting-state connectomes; subject i's own scan has a thick frame
    cw, ch = tw * 0.78, th * 0.78
    dx, dy = 0.019, -0.058
    cx = 0.195
    cy = ymid + 1.5 * abs(dy) - ch / 2
    others = [j for j in range(n) if j != i][:3]
    order = others[:2] + [i] + others[2:]
    for k, j in enumerate(order):
        own = j == i
        thumb(fig, [cx + k * dx, cy + k * dy, cw, ch], fc("rest", j), None, frame=INK if own else MUTED,
              lw=1.8 if own else 0.5)
    cmid = cx + 1.5 * dx + cw / 2
    fig.text(cmid, y_title, "Catalog: rest scans\nof all $N=%d$" % n, ha="center", va="bottom", fontsize=FS,
             linespacing=1.1)
    fig.text(cmid, y_label, r"$C_j^{\mathrm{rest}}$", ha="center", va="top", fontsize=8)
    arrow(fig, (px + tw + 0.008, ymid), (cx - 0.008, ymid))

    # similarity profiles of this subject, annotated with the cohort-level median
    ax_x, ax_w = 0.445, 0.305
    arrow(fig, (cx + 3 * dx + cw + 0.008, ymid), (ax_x - 0.012, ymid))
    fig.text(ax_x + ax_w / 2, 0.985, r"$r_{ij}=\mathrm{corr}(C_i^{\mathrm{task}},C_j^{\mathrm{rest}})$",
             ha="center", va="top", fontsize=8)
    rows = [(sim_raw[i], "raw connectome", MUTED, z_raw, 0.555), (sim_ref[i], "ConvAE+SDL", BLUE, z_ref, 0.155)]
    for prof, lab, col, zmed, yb in rows:
        ax = fig.add_axes([ax_x, yb, ax_w, 0.285])
        lo = prof.min() - 0.02 * np.ptp(prof)
        ax.vlines(np.arange(n), lo, prof, color="#c3cad2", lw=0.33)
        ax.vlines([i], lo, [prof[i]], color=col, lw=1.4)
        ax.plot([i], [prof[i]], "o", ms=2.6, color=col)
        ax.axhline(np.max(np.delete(prof, i)), color=INK, lw=0.5, ls=(0, (2, 2)))
        ax.set_xlim(-4, n + 3)
        ax.set_ylim(lo, prof.max() + 0.42 * np.ptp(prof))
        ax.set_yticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.set_xticks([0, i, n - 1])                       # the highlighted stem sits at j = i
        ax.set_xticklabels(["1", "$i$", str(n)])
        ax.tick_params(axis="x", pad=1)
        ax.text(0.0, 1.0, lab, transform=ax.transAxes, ha="left", va="top", fontsize=FS_SMALL, color=INK)
        ax.text(1.0, 1.0, "cohort +%.1f SD" % zmed, transform=ax.transAxes, ha="right", va="top",
                fontsize=FS_SMALL, color=INK)
    fig.text(ax_x + ax_w / 2, 0.005, "catalog subject $j$", ha="center", va="bottom", fontsize=FS_SMALL)

    # decision rule
    bx, bw = 0.80, 0.195
    box(fig, [bx, 0.15, bw, 0.68], LIGHT, "#9aa5b1")
    xc = bx + bw / 2
    fig.text(xc, 0.785, "Decision", ha="center", va="top", fontsize=FS, weight="bold")
    fig.text(xc, 0.615, r"$\hat{j}=\arg\max_j\,r_{ij}$", ha="center", va="center", fontsize=8)
    fig.text(xc, 0.455, r"correct if $\hat{j}=i$", ha="center", va="center", fontsize=FS)
    fig.text(xc, 0.275, "Top-1 accuracy:\nshare of subjects\nmatched correctly", ha="center", va="center",
             fontsize=FS_SMALL, linespacing=1.15)
    arrow(fig, (ax_x + ax_w + 0.004, ymid), (bx - 0.004, ymid))
    save(fig, "fig1_fcf_process")
    return {"subject_index": i, "median_self_z_raw": float(z_raw), "median_self_z_refined": float(z_ref)}


# --------------------------------------------------------------- Fig. 2 ----
def fig2():
    """ConvAE + SDL with the true tensor sizes, and real matrices of one subject."""
    art = np.load(os.path.join(ART, "stageA_wm.npz"))
    C, Ch, R, DX, Ct = (art[k].astype(np.float64) for k in ("ex_fc", "ex_recon", "ex_resid", "ex_dx", "ex_refined"))

    W, H = TEXTWIDTH_IN, 2.7
    fig = plt.figure(figsize=(W, H))
    asp = W / H
    tw = 0.1
    th = tw * asp

    # ---------------- row (a): ConvAE
    ya = 0.675
    fig.text(0.0, 0.99, r"(a) Convolutional autoencoder (ConvAE): 33,969 parameters, trained with $\mathcal{L}_{\mathrm{MSE}}$",
             ha="left", va="top", fontsize=FS)
    xl = 0.004                      # keeps the outer frames inside the canvas
    thumb(fig, [xl, ya - th / 2, tw, th], C, None)
    fig.text(xl + tw / 2, ya - th / 2 - 0.02, r"input $C_i$", ha="center", va="top", fontsize=FS)

    blocks = [  # (output size label, channels, spatial size, colour)
        (r"16$\times$180$^2$", 16, 180, BLUE), (r"32$\times$90$^2$", 32, 90, BLUE),
        (r"$\mathbf{z}_i$: 64$\times$45$^2$", 64, 45, PURPLE),
        (r"32$\times$90$^2$", 32, 90, ORANGE), (r"16$\times$180$^2$", 16, 180, ORANGE),
        (r"1$\times$360$^2$", 1, 360, ORANGE),
    ]
    x0 = xl + tw + 0.03
    slot = (1.0 - 2 * (xl + tw) - 0.06) / len(blocks)
    hmax = th * 0.95
    prev_right = xl + tw + 0.004
    for k, (size, ch, sp, col) in enumerate(blocks):
        h = hmax * (0.32 + 0.68 * (np.log2(sp) - np.log2(45)) / (np.log2(360) - np.log2(45)))
        w = 0.012 + 0.028 * (np.log2(max(ch, 1)) / 6.0)
        cxm = x0 + (k + 0.5) * slot
        fig.patches.append(Rectangle((cxm - w / 2, ya - h / 2), w, h, transform=fig.transFigure, fc=col, ec=INK,
                                     lw=0.4, alpha=0.85))
        arrow(fig, (prev_right + 0.004, ya), (cxm - w / 2 - 0.004, ya), lw=0.6, ms=4.5)
        fig.text(cxm, ya - hmax / 2 - 0.02, size, ha="center", va="top", fontsize=FS)
        prev_right = cxm + w / 2
    xr = 1.0 - tw - xl
    arrow(fig, (prev_right + 0.004, ya), (xr - 0.006, ya), lw=0.6, ms=4.5)
    thumb(fig, [xr, ya - th / 2, tw, th], Ch, None)
    fig.text(xr + tw / 2, ya - th / 2 - 0.02, r"output $\hat{C}_i$", ha="center", va="top", fontsize=FS)
    for a, b, lab, col in ((x0, x0 + 3 * slot, "encoder: conv + pool", BLUE),
                           (x0 + 3 * slot, x0 + 6 * slot, "decoder: transposed conv", ORANGE)):
        yy = ya + hmax / 2 + 0.04
        fig.lines.append(plt.Line2D([a + 0.008, b - 0.008], [yy, yy], transform=fig.transFigure, color=col, lw=0.8))
        fig.text((a + b) / 2, yy + 0.015, lab, ha="center", va="bottom", fontsize=FS_SMALL, color=INK)

    # ---------------- row (b): residual and SDL
    yb = 0.215
    fig.text(0.0, 0.43, "(b) Residual, refined by sparse dictionary learning (one dictionary per task)",
             ha="left", va="top", fontsize=FS)
    vr = float(np.percentile(np.abs(R[TRIL]), 99))
    # C_i and C^_i already appear in row (a); row (b) starts from their difference
    rx = 0.03
    thumb(fig, [rx, yb - th / 2, tw, th], R, vr)
    fig.text(rx + tw / 2, yb - th / 2 - 0.02, "residual\n" + r"$R_i=C_i-\hat{C}_i$", ha="center", va="top",
             fontsize=FS, linespacing=1.1)

    sx, sw = 0.215, 0.29
    box(fig, [sx, yb - 0.14, sw, 0.28], LIGHT, "#9aa5b1")
    fig.text(sx + sw / 2, yb + 0.118, "K-SVD dictionary", ha="center", va="top", fontsize=FS, weight="bold")
    fig.text(sx + sw / 2, yb + 0.042, r"$\mathbf{e}_i=\mathrm{vec}_{\mathrm{lt}}(R_i)\in\mathbb{R}^{64{,}620}$",
             ha="center", va="center", fontsize=FS)
    fig.text(sx + sw / 2, yb - 0.025, r"atoms $D\in\mathbb{R}^{64{,}620\times K}$", ha="center", va="center",
             fontsize=FS)
    fig.text(sx + sw / 2, yb - 0.09, r"code $\hat{\mathbf{x}}_i\in\mathbb{R}^{K}$, $\|\hat{\mathbf{x}}_i\|_0\leq L$",
             ha="center", va="center", fontsize=FS)
    arrow(fig, (rx + tw + 0.006, yb), (sx - 0.006, yb), lw=0.6, ms=4.5)

    tx = 0.585
    thumb(fig, [tx, yb - th / 2, tw, th], DX - np.diag(np.diag(DX)), vr)
    fig.text(tx + tw / 2, yb - th / 2 - 0.02, "shared part\n" + r"$\mathrm{mat}(D\hat{\mathbf{x}}_i)$",
             ha="center", va="top", fontsize=FS, linespacing=1.1)
    arrow(fig, (sx + sw + 0.006, yb), (tx - 0.006, yb), lw=0.6, ms=4.5)
    arrow(fig, (tx + tw + 0.006, yb), (xr - 0.006, yb), lw=0.6, ms=4.5)
    fig.text((tx + tw + xr) / 2, yb + 0.025, r"subtract from $R_i$", ha="center", va="bottom", fontsize=FS)
    thumb(fig, [xr, yb - th / 2, tw, th], Ct - np.diag(np.diag(Ct)), vr)
    fig.text(1.0 - xl, yb - th / 2 - 0.02, "fingerprint " + r"$\tilde{C}_i$" + "\n" + r"$360\times360$",
             ha="right", va="top", fontsize=FS, linespacing=1.1, multialignment="center")
    save(fig, "fig2_pipeline")
    return {"residual_scale_p99": vr}


# --------------------------------------------------------------- Fig. 3 ----
def network_index():
    reg = np.load(REGIONS)
    nets = reg[:, 1]
    return [np.flatnonzero(nets == nm) for nm in NET_ORDER]


def block_means(mat, groups):
    k = len(groups)
    out = np.zeros((k, k))
    for a in range(k):
        for b in range(k):
            sub = mat[np.ix_(groups[a], groups[b])]
            if a == b:
                iu = np.triu_indices(len(groups[a]), 1)
                out[a, b] = sub[iu].mean() if len(iu[0]) else np.nan
            else:
                out[a, b] = sub.mean()
    return out


def sym0(v):
    m = np.zeros((P, P))
    m[TRIL] = v
    return m + m.T


def atom_summary(task):
    """Network-level description of the dictionary of one task (Stage A)."""
    art = np.load(os.path.join(ART, "stageA_%s.npz" % task))
    D = art["D"].astype(np.float64)
    X = art["X"].astype(np.float64)
    mres = art["mean_resid"].astype(np.float64)[TRIL]
    groups = network_index()
    energy = np.sum(X ** 2, axis=1)                     # atoms have unit norm
    order = np.argsort(energy)[::-1]
    share = energy[order] / energy.sum()
    atoms = []
    for k in order:
        d = D[:, k] * (1 if np.mean(X[k]) >= 0 else -1)  # orient to the typical sign of its use
        atoms.append({"vec": d, "r_mean_resid": float(np.corrcoef(d, mres)[0, 1]),
                      "use": float(np.mean(X[k] != 0)), "blocks": block_means(sym0(d), groups)})
    return atoms, share


def fig3(task="wm", n_show=3):
    """Network-level view of the strongest dictionary atoms of one task.

    cividis has monotonic lightness, so the sign of a block (dark = negative,
    light = positive, mid-grey = zero) survives black-and-white printing.
    """
    atoms, share = atom_summary(task)
    W, H = TEXTWIDTH_IN, 1.80
    fig = plt.figure(figsize=(W, H))
    left, right = 0.075, 0.905
    gap = 0.045
    aw = (right - left - (n_show - 1) * gap) / n_show
    ah = aw * W / H
    bottom = 0.185
    for k in range(n_show):
        a = atoms[k]
        b = a["blocks"] / np.nanmax(np.abs(a["blocks"]))   # atoms are unit-norm: only the pattern matters
        ax = fig.add_axes([left + k * (aw + gap), bottom, aw, ah])
        im = ax.imshow(b, cmap="cividis", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_xticks(range(12))
        ax.set_yticks(range(12))
        ax.set_xticklabels(NET_SHORT, rotation=90, fontsize=FS_SMALL)
        ax.set_yticklabels(NET_SHORT if k == 0 else [], fontsize=FS_SMALL)
        ax.tick_params(length=0, pad=1.5)
        for s in ax.spines.values():
            s.set_linewidth(0.5)
        ax.set_title("Atom %d: %.0f%% of energy\n$r=%.2f$" % (k + 1, 100 * share[k], a["r_mean_resid"]),
                     fontsize=FS, pad=3, linespacing=1.15)
    cax = fig.add_axes([right + 0.018, bottom, 0.012, ah])
    cb = fig.colorbar(im, cax=cax, ticks=[-1, 0, 1])
    cb.ax.tick_params(labelsize=FS_SMALL, length=2, pad=1.5)
    cb.outline.set_linewidth(0.5)
    save(fig, "fig3_atoms_wm")
    glob = [float(np.mean(a["vec"]) ** 2 / np.mean(a["vec"] ** 2)) for a in atoms]
    return {"share": [float(s) for s in share[:n_show]], "r_mean_resid": [a["r_mean_resid"] for a in atoms[:n_show]],
            "global_fraction": glob[:n_show]}


if __name__ == "__main__":
    style()
    info = {}
    for name in sys.argv[1:] or ["fig1", "fig2", "fig3"]:
        info[name] = globals()[name]()
    print(json.dumps(info, indent=1, default=float))
