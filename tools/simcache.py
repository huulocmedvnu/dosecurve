"""Count-level semi-synthetic Perturb-seq cache for the benchmark grid.

usage: python -I tools/simcache.py <src_cache> <out_dir> --a A --b B --tau T
       [--gamma 1] [--groups 200] [--seed 1] [--eff 0.5] [--ngenes 300] [--minsrc 100]

Held-out control cells (25%) become synthetic perturbed cells; the rest stay
as the control group. Per synthetic cell:
  knockdown k ~ Beta(a, b) (10% escapers, k = 0), clipped to [0, 0.98]
  response h = k^gamma * s, log s ~ N(-tau^2/2, tau^2)   (dose-independent
  cell-to-cell heterogeneity)
  target counts: binomial thinning by (1 - k)
  300 response genes (control mean >= 0.2 UMI), half up and half down:
  mean multiplied by exp(+-eff h); down by thinning, up by adding
  Poisson(mu_g L_i / mean L (f - 1)).
Group names "SIM_i|TARGET" borrow size and target gene of a real group, so
Song PS, Mixscale and the joint model run unchanged. Writes truth.tsv.
"""
import argparse, os, shutil
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--a", type=float, required=True); ap.add_argument("--b", type=float, required=True)
ap.add_argument("--tau", type=float, default=0.0); ap.add_argument("--gamma", type=float, default=1.0)
ap.add_argument("--groups", type=int, default=200); ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--eff", type=float, default=0.5); ap.add_argument("--ngenes", type=int, default=300)
ap.add_argument("--escape", type=float, default=0.1)
ap.add_argument("--minsrc", type=int, default=100)
A = ap.parse_args()
rng = np.random.default_rng(A.seed)
S = A.src
indptr = np.fromfile(f"{S}/indptr.bin", np.uint64)
idx = np.memmap(f"{S}/idx.bin", np.uint16, "r")
cnt = np.memmap(f"{S}/cnt.bin", np.uint16, "r")
lib = np.fromfile(f"{S}/lib.bin", np.float32)
batch = np.fromfile(f"{S}/batch.bin", np.uint16) if os.path.exists(f"{S}/batch.bin") else None
members = np.fromfile(f"{S}/members.bin", np.uint32)
genes = open(f"{S}/genes.txt").read().split("\n")
genes = [g for g in genes if g]
gid = {g: i for i, g in enumerate(genes)}
groups = [l.split("\t") for l in open(f"{S}/groups.tsv").read().strip().split("\n")]
groups = [(n, int(o), int(c)) for n, o, c in groups]
ctrl = members[groups[0][1]:groups[0][1] + groups[0][2]]
hold_mask = rng.random(len(ctrl)) < 0.25
hold, ref = ctrl[hold_mask], ctrl[~hold_mask]

# control mean count per gene (for up-regulation) and mean library
G = len(genes)
mu = np.zeros(G)
for r in ref[:3000]:
    a, b = int(indptr[r]), int(indptr[r + 1])
    np.add.at(mu, idx[a:b], cnt[a:b] / lib[r])
mu /= min(len(ref), 3000)  # mean count per unit library
Lbar = lib[ref].mean()
expressed = np.where(mu * Lbar >= 0.2)[0]

src = [g for g in groups[1:] if g[2] >= A.minsrc and g[0] in gid]
A.ngenes = min(A.ngenes, len(expressed) - 1)
os.makedirs(A.out, exist_ok=True)
rows_idx, rows_cnt, rows_lib, rows_b = [], [], [], []
truth = []


def add_row(ix, ct, b):
    rows_idx.append(ix.astype(np.uint16)); rows_cnt.append(ct.astype(np.uint16))
    rows_lib.append(float(ct.sum())); rows_b.append(b)


for r in ref:
    a, b = int(indptr[r]), int(indptr[r + 1])
    add_row(np.asarray(idx[a:b]), np.asarray(cnt[a:b]), batch[r] if batch is not None else 0)
new_groups = [("control", list(range(len(ref))))]
for gi in range(A.groups):
    name, _, n = src[rng.integers(len(src))]
    n = min(n, len(hold))
    tg = gid[name]
    cells = rng.choice(hold, n, replace=False)
    resp = rng.choice(expressed[expressed != tg], A.ngenes, replace=False)
    sign = np.where(np.arange(A.ngenes) % 2 == 0, 1.0, -1.0)
    mem = []
    fmag = A.eff
    dense = np.zeros(G, np.int64)
    for r in cells:
        k = 0.0 if rng.random() < A.escape else min(rng.beta(A.a, A.b), 0.98)
        s = np.exp(A.tau * rng.standard_normal() - 0.5 * A.tau ** 2) if A.tau > 0 else 1.0
        h = (k ** A.gamma) * s if k > 0 else 0.0
        a, b = int(indptr[r]), int(indptr[r + 1])
        ix = np.asarray(idx[a:b]).astype(np.int64)
        dense[ix] = cnt[a:b]
        if dense[tg] > 0:
            dense[tg] = rng.binomial(dense[tg], 1.0 - k)
        f = np.exp(sign * fmag * h)
        c0 = dense[resp]
        dn = f < 1.0
        c1 = c0.copy()
        c1[dn] = rng.binomial(c0[dn], f[dn])
        c1[~dn] = c0[~dn] + rng.poisson(mu[resp[~dn]] * lib[r] * (f[~dn] - 1.0))
        dense[resp] = c1
        nz = np.flatnonzero(dense)
        add_row(nz, np.minimum(dense[nz], 65535), batch[r] if batch is not None else 0)
        dense[nz] = 0
        mem.append(len(rows_lib) - 1)
        truth.append((f"SIM_{gi}|{name}", len(rows_lib) - 1, k, s))
    new_groups.append((f"SIM_{gi}|{name}", mem))

# write the cache
shutil.copy(f"{S}/genes.txt", f"{A.out}/genes.txt")
ip = np.zeros(len(rows_lib) + 1, np.uint64)
ip[1:] = np.cumsum([len(x) for x in rows_idx])
ip.tofile(f"{A.out}/indptr.bin")
np.concatenate(rows_idx).tofile(f"{A.out}/idx.bin")
np.concatenate(rows_cnt).tofile(f"{A.out}/cnt.bin")
np.array(rows_lib, np.float32).tofile(f"{A.out}/lib.bin")
if batch is not None:
    np.array(rows_b, np.uint16).tofile(f"{A.out}/batch.bin")
flat, off = [], 0
with open(f"{A.out}/groups.tsv", "w") as fh:
    for n, m in new_groups:
        fh.write(f"{n}\t{off}\t{len(m)}\n"); flat += m; off += len(m)
np.array(flat, np.uint32).tofile(f"{A.out}/members.bin")
with open(f"{A.out}/truth.tsv", "w") as fh:
    fh.write("group\trow\tk\ts\n")
    for t in truth:
        fh.write("%s\t%d\t%.4f\t%.4f\n" % t)
print(f"{A.out}: control {len(ref)}, groups {A.groups}, cells {len(rows_lib)}")
