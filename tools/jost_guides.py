"""Guide-level dose response in Jost et al. 2020 (attenuated guide series).

usage: python -I tools/jost_guides.py
For each target gene, cells are grouped by guide. Guide knockdown
k_g = 1 - sum(target UMI) / sum(mu L_i), with mu from control cells
(out/jost/target_mix.tsv); guide response = mean cross-fitted score, scaled
per fold by the mean score of the most active guide. gamma_guide is fitted by
weighted least squares of r_g = A k_g^gamma on a log grid (0.25 to 4), with a
95% interval from 300 bootstrap resamples of cells within guides. If guides
mixed escaping and fully depleted cells, the guide-level relation would be
linear whatever the single-cell curve, so gamma_guide > 1 indicates graded
knockdown with a convex response.
Writes out/jost/guides.tsv and out/jost/gene_gamma.tsv.
"""
import numpy as np, pandas as pd, anndata as ad
h = ad.read_h5ad("ext/jost2020/jost.h5ad", backed="r")
guide_of_row = h.obs.guide.values
mem = np.fromfile("cache/jost/members.bin", np.uint32)
groups = [l.split("\t") for l in open("cache/jost/groups.tsv").read().strip().split("\n")]
tm = pd.read_csv("out/jost/target_mix.tsv", sep="\t").set_index("group")
j = pd.read_csv("out/jost/jost_joint.tsv", sep="\t").set_index("gene")
G = np.exp(np.linspace(np.log(0.25), np.log(4), 81))
rng = np.random.default_rng(1)
rows, genes = [], []


def fit(k, r, w):
    best = (np.inf, None)
    for g in G:
        x = k ** g
        A = (w * x * r).sum() / max((w * x * x).sum(), 1e-12)
        rss = (w * (r - A * x) ** 2).sum()
        if rss < best[0]:
            best = (rss, g)
    return best[1]


for name, off, n in groups:
    if name in ("control", "*") or name not in tm.index:
        continue
    cells = mem[int(off):int(off) + int(n)]
    d = pd.read_csv(f"out/jost/cells/d_{name}.tsv", sep="\t")
    d["guide"] = guide_of_row[cells]
    mu = tm.loc[name, "mu"]
    d["exp"] = mu * d.L
    # per-fold scale: mean score of the guide with the lowest target expression
    agg = d.groupby("guide").agg(n=("u", "size"), cnt=("cnt", "sum"), exp=("exp", "sum"))
    agg["k"] = 1 - agg.cnt / agg.exp
    top = agg.k.idxmax()
    for f in (0, 1):
        sc = d[(d.guide == top) & (d.fold == f)].u.mean()
        d.loc[d.fold == f, "r"] = d.loc[d.fold == f, "u"] / sc
    agg["r"] = d.groupby("guide").r.mean()
    agg["se"] = d.groupby("guide").r.std() / np.sqrt(agg.n)
    agg = agg[agg.n >= 20]
    k, r, w = agg.k.clip(0, 1).values, agg.r.values, 1 / np.maximum(agg.se.values, 1e-3) ** 2
    g_hat = fit(k, r, w)
    boot = []
    by = {g: x for g, x in d.groupby("guide") if g in agg.index}
    for _ in range(300):
        kk, rr, ww = [], [], []
        for g, x in by.items():
            b = x.sample(len(x), replace=True, random_state=int(rng.integers(1 << 31)))
            kk.append(1 - b.cnt.sum() / b.exp.sum()); rr.append(b.r.mean()); ww.append(w[list(agg.index).index(g)])
        boot.append(fit(np.clip(kk, 0, 1), np.array(rr), np.array(ww)))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    cls = "buffered" if lo > 1 else ("sensitive" if hi < 1 else "indeterminate")
    genes.append(dict(gene=name, n_guides=len(agg), k_min=k.min(), k_max=k.max(), gamma_guide=g_hat, lo=lo, hi=hi, class_guide=cls,
                      tmean=tm.loc[name, "tmean"], em_class=j.loc[name, "class"], em_gamma=j.loc[name, "gamma"]))
    for gname, a in agg.iterrows():
        rows.append(dict(gene=name, guide=gname, n=a.n, k=a.k, r=a.r, se=a.se))
gg = pd.DataFrame(genes)
pd.DataFrame(rows).to_csv("out/jost/guides.tsv", sep="\t", index=False, float_format="%.4g")
gg.to_csv("out/jost/gene_gamma.tsv", sep="\t", index=False, float_format="%.3g")
pd.set_option("display.width", 200)
print(gg.round(2).to_string(index=False))
