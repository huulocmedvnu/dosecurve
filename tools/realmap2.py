"""Identifiability map with an effect-strength dimension.

usage: python -I tools/realmap2.py [joint.tsv song.genes.tsv mix.tsv out.tsv]
Simulation dirs from env SIMMED (default out/simgrid) and SIMEFF (out/simgrid_eff).
Synthetic groups at true gamma = 1 from three effect levels:
  weak   eff 0.3, 300 response genes   (out/simgrid_eff, weak_*)
  medium eff 0.5, 300 response genes   (out/simgrid, EM run with bootstrap)
  strong eff 0.3, 2500 response genes  (out/simgrid_eff, strong_*)
A setting (effect x dose x tau) is safe when the EM false nonlinear rate on
seed 1 is <= 0.10. A classifier on pipeline features (dose_mid, tau2_hat,
mean_k, target control mean, n cells, phenotype T, DE gene count) is trained
on seed 1 and tested on seed 2, then applied to real groups. Bimodal-dose
settings (40% escapers + Beta(40,1)) are held out to test the map.
Writes out/real/regime2.tsv and out/simgrid_eff/fpr.tsv.
"""
import glob, os, sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

A = sys.argv[1:] if len(sys.argv) > 4 else ["out/real/ess_joint.tsv", "out/real/ess_song.genes.tsv", "out/real/ess_mix.tsv", "out/real/regime2.tsv"]
MED, EFF = os.environ.get("SIMMED", "out/simgrid"), os.environ.get("SIMEFF", "out/simgrid_eff")
FEAT = ["dose_mid", "ltau", "mean_k", "ltmean", "ln", "lT", "lde"]
NL = ["buffered", "sensitive"]


def feats(d):
    d = d.copy()
    d["ltau"] = np.log(d.tau2.clip(lower=0) + 1e-3)
    d["ltmean"] = np.log(d.target_ctrl_mean + 1e-2)
    d["ln"] = np.log(d.n_cells)
    d["lT"] = np.log(d["T"].clip(lower=0.1))
    d["lde"] = np.log1p(d.n_de_lfsr)
    return d


def load(f, eff, dose, tau, seed):
    d = feats(pd.read_csv(f, sep="\t")).set_index("gene")
    base = f[:-len("_joint.tsv")]
    for meth, suf in [("song", "_song.genes.tsv"), ("mix", "_mix.tsv")]:
        x = pd.read_csv(base + suf, sep="\t").set_index("gene")["class"] if os.path.exists(base + suf) else pd.Series(dtype=str)
        d[meth] = x.reindex(d.index).fillna("NA")
    d["em"] = d["class"].fillna("NA")
    d = d[(d.status == "ok") & (d.pheno == 1)].reset_index()
    d["eff"], d["dose"], d["tau"], d["seed"] = eff, dose, float(tau), seed
    return d


rows = []
for f in glob.glob(f"{MED}/g1_*_joint.tsv"):
    _, dose, t, s = os.path.basename(f)[:-10].split("_")
    rows.append(load(f, "medium", dose, t[1:], s[1:]))
for f in glob.glob(f"{EFF}/*_g1_*_joint.tsv"):
    tag = os.path.basename(f)[:-10]
    if tag.startswith("bimod"):
        esc, lev, _, t, s = tag.split("_")
        rows.append(load(f, {"med": "medium", "str": "strong"}[lev], esc, t[1:], s[1:]))
        continue
    eff, _, dose, t, s = tag.split("_")
    rows.append(load(f, eff, dose, t[1:], s[1:]))
S = pd.concat(rows).dropna(subset=FEAT)
S["false"] = S.em.isin(NL)
S["edge"] = S.gamma >= 3.9

# false call table per setting
key = ["eff", "dose", "tau"]
tab = S.groupby(key + ["seed"]).apply(lambda x: pd.Series(dict(
    n=len(x), song=x.song.isin(NL).mean(), mix=x.mix.isin(NL).mean(), em=x.em.isin(NL).mean(),
    em_buf=(x.em == "buffered").mean(), edge_in_false=x.edge[x.false].mean() if x.false.any() else np.nan,
    T=x["T"].median(), nde=x.n_de_lfsr.median())), include_groups=False).reset_index()
tab.to_csv(f"{EFF}/fpr.tsv", sep="\t", index=False)
safe_set = set(map(tuple, tab[(tab.seed == "1") & (tab.em <= 0.10)][key].values))
S["safe"] = [int((e, d, t) in safe_set) for e, d, t in S[key].values]
pd.set_option("display.width", 200)
print("seed 2 false nonlinear rate at gamma = 1 (EM buffered share, share of EM false calls at gamma 4)")
t2 = tab[tab.seed == "2"].sort_values(key)
for _, r in t2.iterrows():
    print(f"{r.eff:6s} {r.dose:7s} tau {r.tau:.1f} | T {r['T']:5.1f} DE {r.nde:5.0f} | Song {r.song:.2f} Mix {r.mix:.2f} EM {r.em:.2f} ({r.em_buf:.2f}, edge {r.edge_in_false:.2f}) | safe {int((r.eff, r.dose, r.tau) in safe_set)}")
print("\nby effect level, seed 2:", t2.groupby("eff")[["song", "mix", "em"]].mean().round(2).to_dict("index"))

tr, te = S[S.seed == "1"], S[S.seed == "2"]
clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=1).fit(tr[FEAT], tr.safe)
te = te.assign(p=clf.predict_proba(te[FEAT])[:, 1])
print(f"\nclassifier: train {len(tr)}, test AUC {roc_auc_score(te.safe, te.p):.3f}")
if os.environ.get("PRED"):
    te[["eff", "dose", "tau", "safe", "p", "false", "em", "song", "mix", "gamma"]].to_csv(os.environ["PRED"], sep="\t", index=False)
for thr in (0.5, 0.8):
    k = te[te.p >= thr]
    print(f"  P >= {thr}: kept {len(k)/len(te):.2f}, truly safe {k.safe.mean():.2f}, EM false rate in kept {k.false.mean():.3f}, Song {k.song.isin(NL).mean():.3f}, Mix {k.mix.isin(NL).mean():.3f}")

# bimodal settings (in training): how often seed 2 groups are mapped safe
B = te[te.dose.str.startswith("bimod")]
if len(B):
    for e, x in B.groupby("dose"):
        k = x[x.p >= 0.5]
        print(f"  {e}: n {len(x)}, EM false {x.false.mean():.2f}, mapped safe {len(k)/len(x):.2f}, EM false among mapped safe {k.false.mean() if len(k) else float('nan'):.2f}")
    k = te[(te.p >= 0.5) & ~te.dose.str.startswith("bimod")]
    print(f"  non-bimodal kept: EM false {k.false.mean():.3f}")

# real data
j = feats(pd.read_csv(A[0], sep="\t")).set_index("gene")
r = j[(j.status == "ok") & (j.pheno == 1)].dropna(subset=FEAT).copy()
r["song"] = pd.read_csv(A[1], sep="\t").set_index("gene")["class"].reindex(r.index).fillna("NA")
r["mix"] = pd.read_csv(A[2], sep="\t").set_index("gene")["class"].reindex(r.index).fillna("NA")
r["em"] = r["class"].fillna("NA")
r["p_safe"] = clf.predict_proba(r[FEAT])[:, 1]
lo, hi = S[FEAT].quantile(0.01), S[FEAT].quantile(0.99)
med = S[S.eff == "medium"][FEAT]
ood = lambda q: ((r[FEAT] < q[0]) | (r[FEAT] > q[1])).any(axis=1).mean()
print(f"\nreal groups {len(r)}: outside sim 1-99% range on any feature: medium only {ood((med.quantile(.01), med.quantile(.99))):.2f}, all levels {ood((lo, hi)):.2f}")
print("  per feature (all levels):", ((r[FEAT] < lo) | (r[FEAT] > hi)).mean().round(2).to_dict())
r[["n_cells", "T", "n_de_lfsr", "target_ctrl_mean", "dose_mid", "tau2", "mean_k", "p_safe", "song", "mix", "em", "gamma"]].to_csv(A[3], sep="\t")
for thr in (0.5, 0.8):
    print(f"  P(safe) >= {thr}: {(r.p_safe >= thr).sum()} ({(r.p_safe >= thr).mean():.2f})")
sf = r[r.p_safe >= 0.5].copy()
print(f"\nsafe regime n = {len(sf)}")
for m in ["song", "mix", "em"]:
    print(f"  {m:5s} buffered {(sf[m]=='buffered').mean():.2f} sensitive {(sf[m]=='sensitive').mean():.2f}")
sf["Tq"] = pd.qcut(sf["T"], 3, labels=["low", "mid", "high"])
print("  by T tertile:")
print(sf.groupby("Tq", observed=True).agg(n=("em", "size"), T=("T", "median"), em_buf=("em", lambda x: (x == "buffered").mean()),
      edge=("gamma", lambda x: (x >= 3.9).mean()), song_sen=("song", lambda x: (x == "sensitive").mean()),
      mix_sen=("mix", lambda x: (x == "sensitive").mean())).round(2).to_string())
