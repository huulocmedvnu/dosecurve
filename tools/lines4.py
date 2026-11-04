"""Four cell lines: safe regime, classes by method, and complexes by TP53 status.

usage: python -I tools/lines4.py
Lines and maps (each with its own simulation grid): K562 (TP53 mutant,
out/real/regime_v4.tsv), Jurkat (TP53 mutant, out/jurkat/regime_native.tsv),
RPE1 (TP53 wild type, out/rpe1/mc50/regime_native.tsv), HepG2 (TP53 wild type,
out/hepg2/regime_native.tsv). Complex shares use all perturbations with a
phenotype. Pooled TP53-mutant vs wild-type comparison by Fisher test.
Writes out/final/lines4_summary.tsv and out/final/lines4_complex.tsv.
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import fisher_exact
sys.path.insert(0, "tools")
LINES = {"K562": ("out/real/regime_v4.tsv", "mut"), "Jurkat": ("out/jurkat/regime_native.tsv", "mut"),
         "RPE1": ("out/rpe1/mc50/regime_native.tsv", "wt"), "HepG2": ("out/hepg2/regime_native.tsv", "wt")}
FAM = {"proteasome": r"^PSM[ABCD]\d", "RNA Pol II": r"^POLR2[A-L]$", "RNA Pol I/III": r"^POLR[13]", "cytosolic ribosome": r"^RP[LS]\d",
       "mito ribosome": r"^MRP[LS]", "CCT chaperonin": r"^CCT\d|^TCP1$", "eIF3": r"^EIF3[A-M]$", "nuclear pore": r"^NUP\d",
       "Mediator": r"^MED\d", "spliceosome SF3/SNRP": r"^SF3[AB]\d|^SNRP", "exosome": r"^EXOSC\d", "MCM/replication": r"^MCM\d|^POLA|^POLE|^PRIM",
       "tRNA synthetase": r"^[A-Z]ARS1?$|^[A-Z]ARS2$"}
R = {k: pd.read_csv(v[0], sep="\t", index_col=0) for k, v in LINES.items() if os.path.exists(v[0])}
rows = []
for k, d in R.items():
    s = d[d.p_safe >= 0.5]
    r = dict(line=k, tp53=LINES[k][1], phenotype=len(d), safe=len(s), safe_frac=len(s) / len(d))
    for m in ["song", "mix", "em"]:
        r[f"{m}_buffered"] = (s[m] == "buffered").mean(); r[f"{m}_sensitive"] = (s[m] == "sensitive").mean()
    rows.append(r)
S = pd.DataFrame(rows)
S.to_csv("out/final/lines4_summary.tsv", sep="\t", index=False, float_format="%.3f")
pd.set_option("display.width", 220)
print(S.round(3).to_string(index=False))
rows = []
for fam, pat in FAM.items():
    r = dict(family=fam)
    pool = {"mut": [0, 0], "wt": [0, 0]}
    for k, d in R.items():
        x = d[d.index.str.match(pat)]
        b = int((x.em == "buffered").sum())
        r[f"{k}_n"], r[f"{k}_buf"] = len(x), (b / len(x) if len(x) else np.nan)
        # share relative to the line-wide buffered share among phenotype groups
        r[f"{k}_rel"] = r[f"{k}_buf"] / max((d.em == "buffered").mean(), 1e-9)
        pool[LINES[k][1]][0] += b; pool[LINES[k][1]][1] += len(x)
    (bm, nm), (bw, nw) = pool["mut"], pool["wt"]
    r["mut_buf"], r["wt_buf"], r["n_mut"], r["n_wt"] = bm / max(nm, 1), bw / max(nw, 1), nm, nw
    r["fisher_p"] = fisher_exact([[bm, nm - bm], [bw, nw - bw]])[1] if nm and nw else np.nan
    rows.append(r)
C = pd.DataFrame(rows)
for k, d in R.items():
    print(f"{k}: EM buffered among all phenotype groups {(d.em == 'buffered').mean():.3f}")
C.to_csv("out/final/lines4_complex.tsv", sep="\t", index=False, float_format="%.3f")
print(C.round(2).to_string(index=False))
