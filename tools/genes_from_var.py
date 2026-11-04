"""Rewrite <cache>/genes.txt with symbols from var/<col> of the source h5ad (old or new format).
usage: python -I tools/genes_from_var.py file.h5ad cache_dir [col=gene_name]"""
import sys, h5py
fn, cache = sys.argv[1], sys.argv[2]; col = sys.argv[3] if len(sys.argv) > 3 else "gene_name"
dec = lambda v: [x.decode() if isinstance(x, bytes) else str(x) for x in v]
with h5py.File(fn, "r") as f:
    d = f[f"var/{col}"]
    if isinstance(d, h5py.Group):
        names = [dec(d["categories"][:])[c] for c in d["codes"][:]]
    elif f"var/__categories/{col}" in f:
        cats = dec(f[f"var/__categories/{col}"][:]); names = [cats[c] for c in d[:]]
    else:
        names = dec(d[:])
old = open(f"{cache}/genes.txt").read().split("\n")
old = [x for x in old if x]
assert len(old) == len(names), (len(old), len(names))
seen = {}
out = []
for n in names:
    k = seen.get(n, 0); seen[n] = k + 1
    out.append(n if k == 0 else f"{n}-{k}")
open(f"{cache}/genes.txt", "w").write("\n".join(out) + "\n")
print(cache, len(out), "genes, duplicates renamed", sum(v > 1 for v in seen.values()))
