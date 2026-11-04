"""Replication of classes between K562 essential and K562 genome-wide screens.

usage: python -I tools/replicate_k562.py
Both screens mapped with the K562 grid (tools/realmap2.py). Genes with a
phenotype in both and P(safe) >= 0.5 in both are compared. Writes
out/gwps/replication.tsv.
"""
import numpy as np, pandas as pd
from scipy.stats import fisher_exact, spearmanr
from sklearn.metrics import cohen_kappa_score
a = pd.read_csv("out/real/regime_v4.tsv", sep="\t", index_col=0)
b = pd.read_csv("out/gwps/regime.tsv", sep="\t", index_col=0)
m = a.join(b, lsuffix="_ess", rsuffix="_gw", how="inner")
print("genes with a phenotype in both screens:", len(m))
for lab, s in [("all shared", m), ("safe in both", m[(m.p_safe_ess >= 0.5) & (m.p_safe_gw >= 0.5)])]:
    out = {}
    for meth in ["em", "song", "mix"]:
        x, y = s[f"{meth}_ess"].fillna("NA"), s[f"{meth}_gw"].fillna("NA")
        bx, by = x == "buffered", y == "buffered"
        sx, sy = x == "sensitive", y == "sensitive"
        out[meth] = dict(n=len(s), kappa=cohen_kappa_score(x, y), buf_rep=by[bx].mean() if bx.any() else np.nan,
                         buf_other=by[~bx].mean(), sen_rep=sy[sx].mean() if sx.any() else np.nan, n_buf=int(bx.sum()), n_sen=int(sx.sum()),
                         OR_buf=fisher_exact([[(bx & by).sum(), (bx & ~by).sum()], [(~bx & by).sum(), (~bx & ~by).sum()]])[0])
    rho = spearmanr(s.gamma_ess, s.gamma_gw).statistic
    print(f"\n{lab}: n {len(s)}, EM gamma Spearman {rho:.2f}")
    print(pd.DataFrame(out).T.round(2).to_string())
    if lab == "safe in both":
        pd.DataFrame(out).T.to_csv("out/gwps/replication.tsv", sep="\t")
