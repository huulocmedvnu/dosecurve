"""Benchmark metrics for Song PS, Mixscale-style and the joint model.

usage: python -I tools/bench_metrics.py <tag>   (reads out/bench/<tag>_*.tsv)
(a) mock NTC groups: fraction with phenotype p < 0.05, fraction classified,
    DE genes per mock group
(b) split halves: Spearman of the dose parameter, Cohen's kappa of classes
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

tag = sys.argv[1]
import os
O = os.environ.get("BENCH", "out/bench")


def kappa(a, b):
    lab = sorted(set(a) | set(b))
    t = pd.crosstab(pd.Categorical(a, lab), pd.Categorical(b, lab)).values.astype(float)
    n = t.sum()
    po = np.trace(t) / n
    pe = (t.sum(0) * t.sum(1)).sum() / n**2
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def load(m, part):
    if m == "song":
        d = pd.read_csv(f"{O}/{tag}_song_{part}.genes.tsv", sep="\t")
        d["cls"] = d["class"].fillna("NA")
        d["dose"] = np.log(d["ed50"].astype(float).clip(lower=1e-6))
        # naive separation z: ps_sep is in control-sd units per cell
        d["p"] = stats.norm.sf(d["ps_sep"] * np.sqrt(d["n_cells"]))
        d["nde"] = d["n_de"]
    else:
        d = pd.read_csv(f"{O}/{tag}_{m}_{part}.tsv", sep="\t")
        d["cls"] = d["class"].fillna("NA")
        if m == "mix":
            d["dose"] = np.log(d["ed50"].astype(float).clip(lower=1e-6))
            d["nde"] = d["n_de"]
        else:
            d["dose"] = np.log(d["gamma"].astype(float))
            d["nde"] = d["n_de_lfsr"]
    return d.set_index("gene")


rows = []
for m, lab in [("song", "Song PS two-step"), ("mix", "Mixscale-style two-step"), ("joint", "Joint cross-fitted")]:
    f = load(m, "full")
    mock = f[f.index.str.startswith("NULL_")]
    real = f[~f.index.str.startswith("NULL_")]
    cl_real = real["cls"].isin(["buffered", "sensitive"])
    r = {
        "method": lab,
        "mock_n": len(mock),
        "mock_p05": (mock["p"] < 0.05).mean(),
        "mock_classified": mock["cls"].isin(["buffered", "sensitive"]).mean(),
        "mock_de_median": mock["nde"].median(),
        "mock_de_mean": mock["nde"].mean(),
        "real_classified": int(cl_real.sum()),
        "real_buffered": int((real["cls"] == "buffered").sum()),
        "real_sensitive": int((real["cls"] == "sensitive").sum()),
        "real_indeterminate": int((real["cls"] == "indeterminate").sum()),
    }
    h0, h1 = load(m, "h0"), load(m, "h1")
    j = h0.join(h1, lsuffix="0", rsuffix="1", how="inner")
    j = j[~j.index.str.startswith("NULL_")]
    both = j[j["cls0"].isin(["buffered", "sensitive"]) & j["cls1"].isin(["buffered", "sensitive"])]
    r["half_both_classified"] = len(both)
    r["half_kappa"] = kappa(both["cls0"], both["cls1"]) if len(both) > 5 else np.nan
    r["half_agree"] = (both["cls0"] == both["cls1"]).mean() if len(both) else np.nan
    # kappa over all 3 states incl. not classified, among genes with >= 200 cells
    jj = j[j["n_cells0"] >= 50]
    s0 = jj["cls0"].where(jj["cls0"].isin(["buffered", "sensitive"]), "none")
    s1 = jj["cls1"].where(jj["cls1"].isin(["buffered", "sensitive"]), "none")
    r["half_kappa_3state"] = kappa(s0, s1)
    ok = both[np.isfinite(both["dose0"]) & np.isfinite(both["dose1"])]
    r["half_spearman_dose"] = stats.spearmanr(ok["dose0"], ok["dose1"])[0] if len(ok) > 5 else np.nan
    rows.append(r)

res = pd.DataFrame(rows).set_index("method").T
pd.set_option("display.width", 200)
print(res.to_string())
res.to_csv(f"{O}/{tag}_metrics.tsv", sep="\t")
