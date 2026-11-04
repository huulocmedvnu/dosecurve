"""Posterior predictive check of binned response for selected real groups.

usage: python -I tools/binned_check.py <safe.tsv> <dump_dir> G1,G2,..
Cells are binned by target UMI (0, 1, 2, >=3). Dose posterior uses counts
only (Poisson, mean tm L/Lbar (1-k), prior on a 51-point k grid by EM).
For gamma 1 and 4, m is fitted by least squares of u on E[k^gamma | counts];
the predicted bin means are compared with the observed means (SE in brackets).
"""
import sys, numpy as np, pandas as pd
from scipy.stats import poisson
s = pd.read_csv(sys.argv[1], sep="\t", index_col=0)
K = np.linspace(0, 0.995, 51)
for g in sys.argv[3].split(","):
    d = pd.read_csv(f"{sys.argv[2]}/d_{g.replace("|", "_")}.tsv", sep="\t")
    tm = s.loc[g, "target_ctrl_mean"]
    lam = tm * (d.L.values / d.L.mean())[:, None] * (1 - K)[None, :]
    lik = poisson.pmf(d.cnt.values[:, None], lam)
    pi = np.full(51, 1 / 51)
    for _ in range(300):
        w = lik * pi; w /= w.sum(1, keepdims=True); pi = w.mean(0)
    b = np.minimum(d.cnt.values, 3)
    u = d.u.values
    obs = [u[b == c].mean() for c in range(4)]
    se = [u[b == c].std() / np.sqrt(max((b == c).sum(), 1)) for c in range(4)]
    out = {}
    for gm in (1, 4):
        x = w @ K ** gm
        m = (x @ u) / (x @ x)
        out[gm] = [m * x[b == c].mean() for c in range(4)]
    sse = {gm: sum(((o - p) / max(e, 1e-9)) ** 2 for o, p, e, c in zip(obs, out[gm], se, range(4)) if (b == c).sum() >= 5) for gm in out}
    print(f"{g} EM {s.loc[g,'em']} gamma {s.loc[g,'gamma']:.1f} T {s.loc[g,'T']:.0f} target UMI {tm:.1f}  n per bin {[int((b==c).sum()) for c in range(4)]}")
    print("  observed " + " ".join(f"{o:6.2f}({e:.2f})" for o, e in zip(obs, se)))
    for gm in (1, 4):
        print(f"  gamma {gm}  " + " ".join(f"{p:6.2f}      " for p in out[gm]) + f" chi2 {sse[gm]:.1f}")
