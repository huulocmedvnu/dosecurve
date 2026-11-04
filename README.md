# dosecurve

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23936124.svg)](https://doi.org/10.5281/zenodo.23936124)

Cross-fitted joint modelling of knockdown and transcriptional response in single-cell CRISPRi perturbation screens.

`dosecurve` labels each perturbed gene as **buffered** (the transcriptional response appears only after expression of the gene has been strongly reduced) or **sensitive** (partial knockdown already produces most of the response), and reports whether the data of that perturbation allow such a label to be estimated reliably. It also contains reimplementations of two two-step methods (Song perturbation score and a Mixscale-style procedure) for side-by-side comparison.

This repository accompanies the manuscript *Calibrated dose-response labels for single-cell CRISPRi perturbations by cross-fitted joint modelling* (Tran, 2026).

## Method in brief

1. **Cross-fitting.** Cells of a perturbation are split into two folds. Gene weights estimated with adaptive shrinkage on one fold score the cells of the other, and the null distribution of scores is taken from control cells that were never used to estimate gene parameters.
2. **Joint dose model.** For each cell, the target transcript count (negative binomial) and the cross-fitted score (normal, with a term for cell-to-cell variation in response strength) are linked through a latent response `h = k^gamma`, where `k` is the knockdown. The model is fitted by EM with SQUAREM acceleration on a profile over `gamma`.
3. **Calibration.** A parametric bootstrap under `gamma = 1` gives a p value. Perturbations with a transcriptional effect are labelled buffered (`gamma > 1`), sensitive (`gamma < 1`) or indeterminate.
4. **Identifiable regime.** Count-level simulations built from the control cells of each screen train a classifier that predicts, from seven features of a perturbation, whether its label is reliable.

## Installation

Requirements: Rust (tested with 1.99), the HDF5 C library (for example `brew install hdf5` or `apt install libhdf5-dev`), and Python 3.12 with the packages in `requirements.txt`.

```bash
cd tools/dosecurve
cargo build --release
cd ../..
pip install -r requirements.txt
```

The binary is `tools/dosecurve/target/release/dosecurve`. Shell scripts use this path unless the environment variable `DOSECURVE` points elsewhere.

## Usage

```bash
B=tools/dosecurve/target/release/dosecurve

# 1. convert an h5ad file with raw counts into a compact cache
$B cache screen.h5ad cache/screen --col perturbation --control control

# 2. two-step methods
$B ps cache/screen out/screen_song --axis lin --bins rank --min-cells 100
$B mixscale cache/screen out/screen_mix --min-cells 100

# 3. joint model (bootstrap with 20 replicates)
DOSE_BOOT=20 $B joint cache/screen out/screen_joint --min-cells 100
```

The joint model writes one row per perturbation with, among others, the phenotype statistic (`T`, `p`, `pheno`), the estimated curvature (`gamma` with profile interval), the bootstrap p value (`p_linear`), the response heterogeneity (`tau2`), the counts-only dose spread (`dose_mid`) and the label (`class`). Add `--null 1000` to append 1,000 groups of held-out control cells, and `--half 0` or `--half 1` to analyse random halves of each perturbation.

Useful environment variables: `DOSE_BOOT` (bootstrap replicates, default 50, 20 used in the manuscript), `DOSE_SEED` (bootstrap seed), `DOSE_TOL` and `DOSE_ITERS` (EM convergence), `DOSE_DUMP=prefix` (write per-cell scores and target counts).

## Data

| Screen | Source |
|---|---|
| K562 essential, K562 genome-wide, RPE1 essential (Replogle et al. 2022) | scPerturb, Zenodo https://doi.org/10.5281/zenodo.13350497 |
| Jurkat and HepG2 essential (Nadig et al. 2025) | GEO GSE264667 |
| Attenuated guide series in K562 (Jost et al. 2020) | GEO GSE132080 |

GEO h5ad files of GSE264667 use an older AnnData layout. Before caching, add a perturbation column and replace Ensembl identifiers by gene symbols:

```bash
python3 tools/h5ad_addcol.py GSE264667_hepg2_raw_singlecell_01.h5ad
$B cache GSE264667_hepg2_raw_singlecell_01.h5ad cache/hepg2 --col perturbation --control control
python3 tools/genes_from_var.py GSE264667_hepg2_raw_singlecell_01.h5ad cache/hepg2
```

The Jost data are converted from the 10x matrix with `tools/jost_prep.py ext/jost2020 ext/jost2020/jost.h5ad`.

## Reproducing the analyses

| Step | Script |
|---|---|
| Count-level simulations (one setting) | `tools/simcache.py`, `tools/simone_line.sh` |
| Simulation grids | `configs/k562_grid.txt` (K562, 100 cells), `configs/line_grid.txt` (RPE1, Jurkat, HepG2, 50 cells) |
| Full analysis of one cell line (methods, target-UMI test, grid, map) | `tools/line_pipeline.sh name cache/name 50` |
| Identifiability map and false-call tables | `tools/realmap2.py` |
| Mock groups and split halves | `tools/bench_v4.sh`, `tools/bench_metrics.py` |
| Sensitive, linear and buffered truths in identifiable settings | `tools/gtest_one.sh` with `configs/gamma_test.txt`, `tools/gtest_summary.py` |
| Target-UMI test of graded against all-or-none knockdown | `tools/target_mix.py` |
| Binned response check for single genes | `tools/binned_check.py` |
| Attenuated guide analysis | `tools/jost_prep.py`, `tools/jost_guides.py` |
| Replication between K562 screens | `tools/replicate_k562.py` |
| Cell-line and complex comparisons | `tools/crossline.py`, `tools/lines4.py` |
| Summary tables and figures | `tools/master_table.py`, `tools/figures.py`, `tools/flowchart.py` |

Example for the K562 grid:

```bash
awk '{print "cache/k562ess out/simgrid_k562 100", $0}' configs/k562_grid.txt | xargs -P 3 -L 1 tools/simone_line.sh
SIMMED=out/simgrid_k562 SIMEFF=out/simgrid_k562 python3 tools/realmap2.py \
  out/real/ess_joint.tsv out/real/ess_song.genes.tsv out/real/ess_mix.tsv out/real/regime.tsv
```

Scripts expect to be run from the repository root and write to `out/`. Analyses in the manuscript were run on an Apple M3 Pro laptop (11 CPU cores, 18 GB memory). The joint model analyses a screen of about 2,000 perturbations, including 1,000 mock groups, in about 2 min with a peak memory below 1 GB.

## License

MIT, see `LICENSE`.

## Citation

Tran THL. Calibrated dose-response labels for single-cell CRISPRi perturbations by cross-fitted joint modelling. Manuscript, 2026.

Software: Tran THL. dosecurve v1.0.0. Zenodo. 2026. https://doi.org/10.5281/zenodo.23936124
