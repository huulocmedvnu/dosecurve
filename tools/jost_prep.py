"""Jost et al. 2020 (GSE132080) 10X matrix -> h5ad for dosecurve cache.

usage: python -I tools/jost_prep.py ext/jost2020 out.h5ad
Cells with a single confidently assigned guide (good_coverage, number_of_cells
== 1) are kept. obs: perturbation (target gene, 'control' for non-targeting),
guide, gemgroup. var index: gene symbols made unique.
"""
import sys, gzip, numpy as np, pandas as pd, scipy.io, scipy.sparse as sp, anndata as ad
D, OUT = sys.argv[1], sys.argv[2]
X = scipy.io.mmread(gzip.open(f"{D}/GSE132080_10X_matrix.mtx.gz")).tocsr().T.tocsr().astype(np.int32)
bc = pd.read_csv(f"{D}/GSE132080_10X_barcodes.tsv.gz", header=None)[0].values
genes = pd.read_csv(f"{D}/GSE132080_10X_genes.tsv.gz", sep="\t", header=None)
ids = pd.read_csv(f"{D}/GSE132080_cell_identities.csv.gz").set_index("cell_barcode")
ids = ids[(ids.good_coverage) & (ids.number_of_cells == 1)]
keep = np.isin(bc, ids.index)
X, bc = X[keep], bc[keep]
g = ids.loc[bc, "guide_identity"].values
pert = np.where(pd.Series(g).str.startswith("neg_ctrl"), "control", pd.Series(g).str.split("_").str[0])
obs = pd.DataFrame({"perturbation": pd.Categorical(pert), "guide": g, "gemgroup": ids.loc[bc, "gemgroup"].values.astype(np.int64)}, index=bc)
var = pd.DataFrame(index=pd.Index(genes[1].values).astype(str))
var.index = pd.Index(ad.utils.make_index_unique(var.index))
ad.AnnData(X=X, obs=obs, var=var).write_h5ad(OUT)
print(OUT, X.shape, obs.perturbation.value_counts().head(3).to_dict(), "groups", obs.perturbation.nunique())
