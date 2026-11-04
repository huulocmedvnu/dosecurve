"""Add obs/perturbation as plain strings to an old-format h5ad (codes + obs/__categories).
usage: python -I tools/h5ad_addcol.py file.h5ad [source_col=gene] [control=non-targeting]
Writes 'control' for the control label."""
import sys, h5py, numpy as np
fn = sys.argv[1]; col = sys.argv[2] if len(sys.argv) > 2 else "gene"; ctl = sys.argv[3] if len(sys.argv) > 3 else "non-targeting"
with h5py.File(fn, "r+") as f:
    cats = [c.decode() if isinstance(c, bytes) else c for c in f[f"obs/__categories/{col}"][:]]
    lab = np.array(["control" if c == ctl else c for c in cats], dtype=object)[f[f"obs/{col}"][:]]
    if "perturbation" in f["obs"]:
        del f["obs/perturbation"]
    f["obs"].create_dataset("perturbation", data=lab, dtype=h5py.string_dtype())
    print(fn, "cells", len(lab), "groups", len(set(lab)))
