"""Summary of out/gtest: call rates by method at gamma 0.5, 1, 2 (seed 2, medium effect).
usage: python -I tools/gtest_summary.py"""
import glob, os, pandas as pd
rows = []
for f in sorted(glob.glob("out/gtest/*_v4.tsv")):
    t = os.path.basename(f)[:-7]; g, dose, tau = t.split("_")
    v = pd.read_csv(f, sep="\t").set_index("gene"); a = pd.read_csv(f"out/gtest/{t}_anch.tsv", sep="\t").set_index("gene")
    s = pd.read_csv(f"out/gtest/{t}_song.genes.tsv", sep="\t").set_index("gene")["class"].reindex(v.index)
    m = pd.read_csv(f"out/gtest/{t}_mix.tsv", sep="\t").set_index("gene")["class"].reindex(v.index)
    ph = v.pheno == 1
    r = dict(gamma=g[1:], dose=dose, tau=tau[1:], n=int(ph.sum()))
    for lab, c in [("song", s), ("mix", m), ("v4", v["class"]), ("anch", a["class"].reindex(v.index))]:
        c = c[ph]
        r[f"{lab}_S"], r[f"{lab}_B"] = (c == "sensitive").mean(), (c == "buffered").mean()
    rows.append(r)
d = pd.DataFrame(rows)
d.to_csv("out/gtest/summary.tsv", sep="\t", index=False, float_format="%.3f")
pd.set_option("display.width", 200)
print(d.round(2).to_string(index=False))
