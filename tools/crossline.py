"""K562 vs RPE1: safe-regime fraction, method classes and gene-level concordance.

usage: python -I tools/crossline.py <k562_regime.tsv> <rpe1_regime.tsv> <k562_tmix.tsv> <rpe1_tmix.tsv> <out_prefix>
Regime tables come from tools/realmap2.py (same simulation-trained map),
target-UMI tables from tools/target_mix.py. Writes <out>_genes.tsv and
<out>_complex.tsv.
"""
import sys, numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr

NL = ["buffered", "sensitive"]
FAM = {"proteasome": r"^PSM[ABCD]\d", "RNA Pol II": r"^POLR2[A-L]$", "RNA Pol I/III": r"^POLR[13]", "cytosolic ribosome": r"^RP[LS]\d",
       "mito ribosome": r"^MRP[LS]", "CCT chaperonin": r"^CCT\d|^TCP1$", "eIF3": r"^EIF3[A-M]$", "nuclear pore": r"^NUP\d",
       "Mediator": r"^MED\d", "spliceosome SF3/SNRP": r"^SF3[AB]\d|^SNRP", "exosome": r"^EXOSC\d", "MCM/replication": r"^MCM\d|^POLA|^POLE|^PRIM",
       "tRNA synthetase": r"^[A-Z]ARS1?$|^[A-Z]ARS2$"}


def load(reg, tm):
    r = pd.read_csv(reg, sep="\t", index_col=0)
    t = pd.read_csv(tm, sep="\t").set_index("group")
    r = r.join(t[["tmean", "llr", "beta_mean"]].rename(columns={"tmean": "tm_ctrl"}))
    r["graded"] = (2 * r.llr > 3.84) & (r.tm_ctrl >= 0.5)
    r["safe"] = r.p_safe >= 0.5
    return r


K, R = load(sys.argv[1], sys.argv[3]), load(sys.argv[2], sys.argv[4])
out = sys.argv[5]
for lab, d in [("K562", K), ("RPE1", R)]:
    s = d[d.safe]
    print(f"{lab}: phenotype groups {len(d)}, safe {len(s)} ({len(s)/len(d):.2f}), P >= 0.8 {(d.p_safe >= 0.8).mean():.2f}")
    for m in ["song", "mix", "em"]:
        print(f"  {m:5s} safe: buffered {(s[m]=='buffered').mean():.2f} sensitive {(s[m]=='sensitive').mean():.2f} | all: buffered {(d[m]=='buffered').mean():.2f} sensitive {(d[m]=='sensitive').mean():.2f}")
    b = s[(s.em == "buffered") & (s.tm_ctrl >= 0.5)]
    print(f"  EM buffered in safe with graded target knockdown: {b.graded.mean():.2f} (n {len(b)})")

both = K.join(R, lsuffix="_k", rsuffix="_r", how="inner")
print(f"\ngenes with a phenotype in both lines: {len(both)}")
for lab, sub in [("all shared", both), ("safe in both", both[both.safe_k & both.safe_r])]:
    if len(sub) < 5:
        continue
    kap = cohen_kappa_score(sub.em_k, sub.em_r)
    rho = spearmanr(sub.gamma_k, sub.gamma_r).statistic
    bk, br = sub.em_k == "buffered", sub.em_r == "buffered"
    print(f"  {lab}: n {len(sub)}, EM kappa {kap:.2f}, gamma spearman {rho:.2f}, buffered K {bk.mean():.2f} R {br.mean():.2f}, "
          f"both {(bk & br).sum()}, P(buffered R | buffered K) {br[bk].mean():.2f} vs {br[~bk].mean():.2f}")
    print("   ", pd.crosstab(sub.em_k, sub.em_r).to_dict())
both.to_csv(f"{out}_genes.tsv", sep="\t")

rows = []
for fam, pat in FAM.items():
    for lab, d in [("K562", K), ("RPE1", R)]:
        x = d[d.index.str.match(pat)]
        if len(x) == 0:
            continue
        s = x[x.safe]
        rows.append(dict(family=fam, line=lab, n=len(x), em_buffered=(x.em == "buffered").mean(), n_safe=len(s),
                         em_buffered_safe=(s.em == "buffered").mean() if len(s) else np.nan,
                         song_sensitive=(x.song == "sensitive").mean(), gamma_median=x.gamma.median()))
C = pd.DataFrame(rows)
C.to_csv(f"{out}_complex.tsv", sep="\t", index=False)
print("\ncomplexes (EM buffered share, all phenotype groups; safe in brackets)")
for fam, g in C.groupby("family", sort=False):
    print(f"  {fam:22s} " + " | ".join(f"{r.line} n {r.n:3d} buf {r.em_buffered:.2f} ({r.em_buffered_safe:.2f}, n {r.n_safe})" for r in g.itertuples()))
