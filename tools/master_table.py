"""Master summary tables for Figures 1 to 4 (v4 joint model, B = 20).

usage: python -I tools/master_table.py
Reads out/simgrid_v4, out/simgrid_eff_v4 (fpr.tsv from tools/realmap2.py),
out/real/realmap_v4.log, out/rpe1/realmap_v4.log, out/final/crossline_*.tsv,
regime and target-UMI tables. Writes out/final/fig1_fpr.tsv, fig2_power.tsv,
fig3_map.tsv, fig4_real.tsv, fig4_complex.tsv and out/final/summary.md.
"""
import glob, os, re
import numpy as np, pandas as pd

NL = ["buffered", "sensitive"]
F = "out/final"
fpr = pd.read_csv("out/simgrid_eff_v4/fpr.tsv", sep="\t")
fpr = fpr[fpr.seed == 2].drop(columns="seed")
fpr["dose"] = fpr.dose.replace({"spread": "Beta(2,2)", "moder": "Beta(4,1.5)", "high": "Beta(15,1.5)", "extreme": "Beta(40,1)"})
fpr.to_csv(f"{F}/fig1_fpr.tsv", sep="\t", index=False, float_format="%.3f")

# power at gamma 0.5 and 2 (medium effect, tau 0.3, seed 2)
rows = []
for f in sorted(glob.glob("out/simgrid_v4/g[02]*_joint.tsv")):
    tag = os.path.basename(f)[:-10]
    g, dose, tau, s = tag.split("_")
    want = {"g0.5": "sensitive", "g2": "buffered"}[g]
    j = pd.read_csv(f, sep="\t").set_index("gene")
    alt = lambda x: x if os.path.exists(x) else x.replace("simgrid_v4", "simgrid_rc")
    so = pd.read_csv(alt(f.replace("_joint", "_song.genes")), sep="\t").set_index("gene")["class"].reindex(j.index)
    mi = pd.read_csv(alt(f.replace("_joint", "_mix")), sep="\t").set_index("gene")["class"].reindex(j.index)
    rows.append(dict(gamma=g[1:], dose=dose, tau=tau[1:], song=(so == want).mean(), mix=(mi == want).mean(), em=(j["class"] == want).mean(),
                     em_wrong=(j["class"] == ({"sensitive": "buffered", "buffered": "sensitive"}[want])).mean()))
pw = pd.DataFrame(rows)
pw.to_csv(f"{F}/fig2_power.tsv", sep="\t", index=False, float_format="%.3f")


def grab(log):
    t = open(log).read()
    d = {}
    d["auc"] = float(re.search(r"test AUC ([\d.]+)", t).group(1))
    for thr in ("0.5", "0.8"):
        m = re.search(rf"P >= {thr}: kept ([\d.]+), truly safe ([\d.]+), EM false rate in kept ([\d.]+), Song ([\d.]+), Mix ([\d.]+)", t)
        d.update({f"kept_{thr}": float(m.group(1)), f"truly_safe_{thr}": float(m.group(2)), f"em_fpr_{thr}": float(m.group(3)),
                  f"song_fpr_{thr}": float(m.group(4)), f"mix_fpr_{thr}": float(m.group(5))})
    for e in ("0.2", "0.4", "0.6"):
        m = re.search(rf"bimod{e}: n \d+, EM false ([\d.]+), mapped safe ([\d.]+)", t)
        d[f"bimodal{e}_em_false"], d[f"bimodal{e}_mapped_safe"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"outside sim 1-99% range on any feature: medium only ([\d.]+), all levels ([\d.]+)", t)
    d["ood_medium_only"], d["ood_all"] = float(m.group(1)), float(m.group(2))
    return d


mp = grab("out/real/realmap_v4.log")
mp_r = grab("out/rpe1/mc50/realmap_native.log")
mapt = pd.DataFrame({"K562 (K562 grid)": mp, "RPE1 (RPE1 grid, min cells 50)": mp_r})
mapt.to_csv(f"{F}/fig3_map.tsv", sep="\t", float_format="%.3f")
gt = pd.read_csv("out/gtest/summary.tsv", sep="\t")
gt.to_csv(f"{F}/fig2_gamma_safe.tsv", sep="\t", index=False, float_format="%.3f")

rows = []
for lab, reg, tm, ood in [("K562", "out/real/regime_v4.tsv", "out/real/target_mix.tsv", mp["ood_all"]), ("RPE1", "out/rpe1/mc50/regime_native.tsv", "out/rpe1/target_mix.tsv", mp_r["ood_all"])]:
    r = pd.read_csv(reg, sep="\t", index_col=0)
    t = pd.read_csv(tm, sep="\t").set_index("group")
    r = r.join(t[["tmean", "llr"]])
    s = r[r.p_safe >= 0.5]
    b = s[(s.em == "buffered") & (s.tmean >= 0.5)]
    d = dict(line=lab, phenotype_groups=len(r), safe=len(s), safe_frac=len(s) / len(r), safe_frac_p08=(r.p_safe >= 0.8).mean(), outside_sim_range=ood)
    for m in ["song", "mix", "em"]:
        d[f"{m}_buffered_safe"] = (s[m] == "buffered").mean()
        d[f"{m}_sensitive_safe"] = (s[m] == "sensitive").mean()
    d["em_buffered_graded_target"] = (2 * b.llr > 3.84).mean()
    d["em_buffered_testable"] = len(b)
    s = s.assign(Tq=pd.qcut(s["T"], 3, labels=["low", "mid", "high"]))
    for q, x in s.groupby("Tq", observed=True):
        d[f"em_buffered_T_{q}"] = (x.em == "buffered").mean()
    rows.append(d)
real = pd.DataFrame(rows).set_index("line").T
real.to_csv(f"{F}/fig4_real.tsv", sep="\t", float_format="%.3f")
cx = pd.read_csv(f"{F}/crossline_complex.tsv", sep="\t")
cx.to_csv(f"{F}/fig4_complex.tsv", sep="\t", index=False, float_format="%.3f")
g = pd.read_csv(f"{F}/crossline_genes.tsv", sep="\t", index_col=0)

with open(f"{F}/summary.md", "w") as fh:
    fh.write("# Master summary, joint model v4 (SQUAREM EM, bootstrap B = 20)\n\nK562 essential min cells 100, RPE1 essential min cells 50. Seed flip rate of EM classes at B = 20: 53 of 1102 K562 phenotype groups.\n\n")
    fh.write("## Figure 1 and 2: false nonlinear calls at true gamma = 1 (seed 2)\n\n")
    fh.write(fpr[["eff", "dose", "tau", "song", "mix", "em", "em_buf"]].round(3).to_string(index=False) + "\n\n")
    fh.write("Mean over settings by effect tier:\n\n" + fpr.groupby("eff")[["song", "mix", "em"]].mean().round(3).to_string() + "\n\n")
    fh.write("## Figure 2 (supplement): power at tau 0.3 including unsafe Beta(15,1.5)\n\n" + pw.round(3).to_string(index=False) + "\n\n")
    fh.write("## Figure 2: sensitive (S) and buffered (B) call rates at gamma 0.5, 1, 2 in safe settings (medium effect, seed 2; v4 = reference EM, anch = anchored-dose variant, not adopted)\n\n" + gt.round(2).to_string(index=False) + "\n\n")
    fh.write("## Figure 3: identifiability map (each line mapped with a grid built from its own cache)\n\n" + mapt.round(3).to_string() + "\n\n")
    fh.write("## Figure 4: real data\n\n" + real.round(3).to_string() + "\n\n")
    both = g[(g.p_safe_k >= 0.5) & (g.p_safe_r >= 0.5)]
    fh.write(f"Shared phenotype genes {len(g)}, safe in both {len(both)}. EM class cross-tab (rows K562, columns RPE1), safe in both:\n\n")
    fh.write(pd.crosstab(both.em_k, both.em_r).round(3).to_string() + "\n\n")
    fh.write("Complexes:\n\n" + cx.round(3).to_string(index=False) + "\n")
print(open(f"{F}/summary.md").read())
