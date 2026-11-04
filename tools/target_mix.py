"""Target-UMI test: all-or-none vs graded knockdown, on the target gene's own counts.

usage: python -I tools/target_mix.py <cache_dir> <out.tsv> [G1,G2,..|all]
Control cells give the target's NB model: mean mu L_i per cell (mu per unit
library) and size r (MLE). In perturbed cells, counts ~ NB(mu L_i (1 - k), r).
  all-or-none (AoN): k = 0 with weight pi (escapers), else k = k_hi in [0.9, 0.995]
  graded:            k = 0 with weight pi, else k ~ Beta(a, b) on a 100-point grid
LLR = loglik(graded) - loglik(AoN); AoN is the limit of the Beta at k_hi, so
LLR >= 0 up to optimisation error. Also reported: Beta mean and the Beta mass
in [0.2, 0.8] among non-escapers.
"""
import sys, numpy as np, pandas as pd
from scipy.special import gammaln, betaln
from scipy.optimize import minimize, minimize_scalar

S, OUT = sys.argv[1], sys.argv[2]
sel = sys.argv[3] if len(sys.argv) > 3 else "all"
indptr = np.fromfile(f"{S}/indptr.bin", np.uint64)
idx = np.memmap(f"{S}/idx.bin", np.uint16, "r")
cnt = np.memmap(f"{S}/cnt.bin", np.uint16, "r")
lib = np.fromfile(f"{S}/lib.bin", np.float32).astype(float)
mem = np.fromfile(f"{S}/members.bin", np.uint32)
genes = [g for g in open(f"{S}/genes.txt").read().split("\n") if g]
gid = {g: i for i, g in enumerate(genes)}
groups = [l.split("\t") for l in open(f"{S}/groups.tsv").read().strip().split("\n")]
groups = {n: mem[int(o):int(o) + int(c)] for n, o, c in groups}
names = [n for n in groups if n != "control"] if sel == "all" else sel.split(",")
names = [n for n in names if n.split("|")[-1] in gid]
tg = sorted({gid[n.split("|")[-1]] for n in names})


def counts(rows, gset):
    out = np.zeros((len(rows), len(gset)))
    pos = {g: j for j, g in enumerate(gset)}
    gs = np.array(gset)
    for i, r in enumerate(rows):
        a, b = int(indptr[r]), int(indptr[r + 1])
        ix = np.asarray(idx[a:b])
        m = np.isin(ix, gs)
        for g, c in zip(ix[m], np.asarray(cnt[a:b])[m]):
            out[i, pos[g]] = c
    return out


ctrl = groups["control"]
C = counts(ctrl, tg)
Lc = lib[ctrl]


def nb_ll(c, mean, r):
    mean = np.maximum(mean, 1e-12)
    return gammaln(c + r) - gammaln(r) - gammaln(c + 1) + r * np.log(r / (r + mean)) + c * np.log(mean / (r + mean))


ctrl_fit = {}
for j, g in enumerate(tg):
    c = C[:, j]
    mu = c.sum() / Lc.sum()
    f = lambda lr: -nb_ll(c, mu * Lc, np.exp(lr)).sum()
    lr = minimize_scalar(f, bounds=(-3, 6), method="bounded").x
    ctrl_fit[g] = (mu, np.exp(lr))

K = np.linspace(0, 0.995, 100)
rows = []
for n in names:
    g = gid[n.split("|")[-1]]
    cells = groups[n]
    c = counts(cells, [g])[:, 0]
    L = lib[cells]
    mu, r = ctrl_fit[g]
    P = np.exp(nb_ll(c[:, None], mu * L[:, None] * (1 - K)[None, :], r))  # cells x grid
    p0 = P[:, 0]

    def mix(pi, w):  # w: weights over grid for non-escapers
        return np.log(pi * p0 + (1 - pi) * (P @ w) + 1e-300).sum()

    best_aon = -np.inf
    for kh in np.where(K >= 0.9)[0]:
        w = np.zeros(len(K)); w[kh] = 1
        o = minimize_scalar(lambda t: -mix(1 / (1 + np.exp(-t)), w), bounds=(-8, 8), method="bounded")
        if -o.fun > best_aon:
            best_aon, pi_aon, khi = -o.fun, 1 / (1 + np.exp(-o.x)), K[kh]

    def beta_w(la, lb):
        a, b = np.exp(la), np.exp(lb)
        x = np.clip(K, 1e-3, 1 - 1e-3)
        lw = (a - 1) * np.log(x) + (b - 1) * np.log(1 - x)
        w = np.exp(lw - lw.max()); return w / w.sum()

    obj = lambda th: -mix(1 / (1 + np.exp(-th[0])), beta_w(th[1], th[2]))
    best = None
    for st in [(-2, 0.7, 0.7), (-2, 2.5, 0), (-2, 3.5, -1), (-1, 0, 0), (-2, 1.5, 0.4)]:
        o = minimize(obj, st, method="Nelder-Mead", options=dict(maxiter=600, xatol=1e-3, fatol=1e-4))
        if best is None or o.fun < best.fun:
            best = o
    w = beta_w(best.x[1], best.x[2])
    rows.append(dict(group=n, n=len(c), tmean=mu * L.mean(), size=r, zero_frac=(c == 0).mean(),
                     ll_aon=best_aon, pi_aon=pi_aon, k_hi=khi, ll_graded=-best.fun,
                     pi_graded=1 / (1 + np.exp(-best.x[0])), beta_mean=w @ K,
                     beta_sd=np.sqrt(w @ K ** 2 - (w @ K) ** 2), mid_mass=w[(K >= 0.2) & (K <= 0.8)].sum(),
                     llr=-best.fun - best_aon, beta_a=np.exp(best.x[1]), beta_b=np.exp(best.x[2]), mu=mu, Lmean=L.mean()))
pd.DataFrame(rows).to_csv(OUT, sep="\t", index=False, float_format="%.4g")
print(f"{OUT}: {len(rows)} groups")
