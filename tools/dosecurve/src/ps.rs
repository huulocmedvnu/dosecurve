//! Perturbation score (scMAGeCK scmageck_eff_estimate, Song et al. 2025) and
//! Hill LL.4 dose classification, one perturbation per rayon task.
//!
//! Per perturbation P (cells of P plus all control cells):
//!  1. Seurat FindMarkers-like DE (Wilcoxon, log2FC with pseudocount 1,
//!     min.pct 0.01, |log2FC| >= 0.1, relaxed x0.8 up to 3 rounds when
//!     < 10 genes); keep top 500 by p (|z|).
//!  2. Y = log-normalised expression (ln(1 + 1e4 * count / lib)) of target
//!     genes, genes detected in >= 1% cells, clipped at the per-gene 95th
//!     percentile (R type 7) over P + control cells.
//!  3. Ridge (lambda 0.01) of Y on [1, 1{P}] -> beta_neg, beta_p.
//!  4. Per cell x = argmin_{0<=x<=3} ||x beta_p + beta_neg - y||^2, /3,
//!     divided by the max over all cells.
//!  5. Perturbed-gene expression scaled 0 = min, 1 = control median; ten
//!     quantile bins over P cells; LL.4 fit on bin means; ED50 = e.

use crate::hill::{self, Fit};
use anyhow::{Context, Result};
use memmap2::Mmap;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

pub struct Cache {
    pub genes: Vec<String>,
    pub groups: Vec<(String, usize, usize)>,
    members: Mmap,
    indptr: Mmap,
    idx: Mmap,
    cnt: Mmap,
    pub scale: Vec<f32>, // 1e4 / lib
    pub batch: Option<Vec<u16>>,
}

fn map(p: &Path) -> Result<Mmap> {
    let f = File::open(p).with_context(|| format!("open {}", p.display()))?;
    Ok(unsafe { Mmap::map(&f)? })
}

impl Cache {
    pub fn open(dir: &str) -> Result<Self> {
        let d = Path::new(dir);
        let genes = std::fs::read_to_string(d.join("genes.txt"))?.lines().map(String::from).collect();
        let groups = std::fs::read_to_string(d.join("groups.tsv"))?
            .lines()
            .map(|l| {
                let f: Vec<&str> = l.split('\t').collect();
                (f[0].to_string(), f[1].parse().unwrap(), f[2].parse().unwrap())
            })
            .collect();
        let libm = map(&d.join("lib.bin"))?;
        let lib: &[f32] = bytemuck::cast_slice(&libm);
        let scale = lib.iter().map(|&l| if l > 0.0 { 1e4 / l } else { 0.0 }).collect();
        let batch = std::fs::read(d.join("batch.bin")).ok().map(|b| bytemuck::cast_slice::<u8, u16>(&b).to_vec());
        Ok(Cache { genes, groups, members: map(&d.join("members.bin"))?, indptr: map(&d.join("indptr.bin"))?, idx: map(&d.join("idx.bin"))?, cnt: map(&d.join("cnt.bin"))?, scale, batch })
    }
    /// cell rows of group g
    pub fn cells(&self, g: usize) -> &[u32] {
        let m: &[u32] = bytemuck::cast_slice(&self.members);
        let (_, o, n) = self.groups[g];
        &m[o..o + n]
    }
    #[inline]
    pub fn row(&self, c: usize) -> (&[u16], &[u16]) {
        let ip: &[u64] = bytemuck::cast_slice(&self.indptr);
        let (a, b) = (ip[c] as usize, ip[c + 1] as usize);
        let idx: &[u16] = bytemuck::cast_slice(&self.idx);
        let cnt: &[u16] = bytemuck::cast_slice(&self.cnt);
        (&idx[a..b], &cnt[a..b])
    }
}

/// Control cells in gene-major form.
struct Ctrl {
    n: usize,
    rows: Vec<u32>,     // global cell row of each control cell
    off: Vec<usize>,     // per gene offsets into cell/val (n_genes + 1)
    cell: Vec<u32>,      // control-local cell index
    val: Vec<f32>,       // log-normalised value (cell order)
    sorted: Vec<f32>,    // same values sorted ascending per gene
    sum_log: Vec<f64>,   // per gene sum of log values
    sum_lin: Vec<f64>,   // per gene sum of expm1(log) = normalised counts
}

impl Ctrl {
    fn build(c: &Cache, cells: &[u32], n_genes: usize) -> Ctrl {
        let n = cells.len();
        let mut cntg = vec![0usize; n_genes + 1];
        for i in 0..n {
            for &g in c.row(cells[i] as usize).0 {
                cntg[g as usize + 1] += 1;
            }
        }
        for g in 0..n_genes {
            cntg[g + 1] += cntg[g];
        }
        let off = cntg.clone();
        let mut fill = cntg;
        let nnz = off[n_genes];
        let mut cell = vec![0u32; nnz];
        let mut val = vec![0f32; nnz];
        let mut sum_log = vec![0f64; n_genes];
        let mut sum_lin = vec![0f64; n_genes];
        for i in 0..n {
            let s = c.scale[cells[i] as usize];
            let (ix, ct) = c.row(cells[i] as usize);
            for (&g, &k) in ix.iter().zip(ct) {
                let g = g as usize;
                let lin = k as f32 * s;
                let v = lin.ln_1p();
                cell[fill[g]] = i as u32;
                val[fill[g]] = v;
                fill[g] += 1;
                sum_log[g] += v as f64;
                sum_lin[g] += lin as f64;
            }
        }
        let mut sorted = val.clone();
        let mut chunks: Vec<&mut [f32]> = Vec::with_capacity(n_genes);
        let mut rest = sorted.as_mut_slice();
        for g in 0..n_genes {
            let (h, t) = rest.split_at_mut(off[g + 1] - off[g]);
            chunks.push(h);
            rest = t;
        }
        chunks.par_iter_mut().for_each(|s| s.sort_unstable_by(f32::total_cmp));
        Ctrl { n, rows: cells.to_vec(), off, cell, val, sorted, sum_log, sum_lin }
    }
    #[inline]
    fn nz(&self, g: usize) -> &[f32] {
        &self.sorted[self.off[g]..self.off[g + 1]]
    }
}

pub struct Params {
    pub min_cells: usize,
    pub lfc: f64,
    pub min_pct: f64,
    pub max_targets: usize,
    pub min_targets: usize,
    pub lambda: f64,
    pub scale_factor: f64,
    pub use_perturb_exp: bool,
    pub bins: usize,
    pub bin_mode: BinMode,
    pub axis: Axis,
    /// keep only perturbed cells whose hashed row falls in this half (0/1)
    pub half: Option<u32>,
    /// number of pseudo-perturbations drawn from held-out control cells
    pub null: usize,
}

pub fn splitmix(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[derive(Clone, Copy, PartialEq)]
pub enum Axis {
    /// ln(1 + 1e4 count / lib)
    Log,
    /// 1e4 count / lib
    Lin,
    /// linear value z-scored against control cells of the same batch (Replogle 2022)
    Z,
}

#[derive(Clone, Copy, PartialEq)]
pub enum BinMode {
    /// R: cut(x, unique(quantile(x, 0:10/10)), include.lowest = TRUE)
    CutUnique,
    /// equal-size bins after sorting
    Rank,
}

pub struct PertResult {
    pub name: String,
    pub n_cells: usize,
    pub n_de: usize,
    pub n_target: usize,
    pub lfc_used: f64,
    pub ctrl_median: f32,
    pub frac_zero: f64,
    pub own_lfc: f64,
    pub mean_ps: f64,
    /// (mean PS of perturbed cells - mean PS of control cells) / sd of control PS
    pub ps_sep: f64,
    pub fit: Option<Fit>,
    pub fitb: Option<Fit>,
    pub bins: Vec<(f64, f64, usize)>,
    pub status: &'static str,
    pub cells: Vec<(u32, f32, f32)>, // cache cell, x (scaled expr), PS
}

/// R quantile type 7 on a sample given as `z` zeros plus sorted values `a` and `b`.
fn q7_merged(z: usize, a: &[f32], b: &[f32], p: f64) -> f32 {
    let n = z + a.len() + b.len();
    let h = (n - 1) as f64 * p;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    // walk from the top to fetch order statistics lo and hi
    let kth = |k: usize| -> f32 {
        if k < z {
            return 0.0;
        }
        let from_top = n - 1 - k;
        let (mut i, mut j) = (a.len(), b.len());
        let mut v = 0.0;
        for _ in 0..=from_top {
            if i > 0 && (j == 0 || a[i - 1] >= b[j - 1]) {
                i -= 1;
                v = a[i];
            } else {
                j -= 1;
                v = b[j];
            }
        }
        v
    };
    let (vl, vh) = (kth(lo), kth(hi));
    vl + (h - lo as f64) as f32 * (vh - vl)
}

fn lower_bound(s: &[f32], v: f32) -> usize {
    s.partition_point(|&x| x < v)
}
fn upper_bound(s: &[f32], v: f32) -> usize {
    s.partition_point(|&x| x <= v)
}

fn one(c: &Cache, ctrl: &Ctrl, gid: &HashMap<&str, usize>, name: &str, all: &[u32], prm: &Params) -> PertResult {
    let name = &name.to_string();
    let n_full = all.len();
    let kept: Vec<u32>;
    let cells: &[u32] = match prm.half {
        None => all,
        Some(h) => {
            // splitmix-style hash so halves are random with respect to row order
            kept = all
                .iter()
                .cloned()
                .filter(|&r| {
                    let mut z = (r as u64).wrapping_add(0x9E3779B97F4A7C15);
                    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                    ((z ^ (z >> 31)) & 1) as u32 == h
                })
                .collect();
            &kept
        }
    };
    let np = cells.len();
    let ng = c.genes.len();
    let nc = ctrl.n;
    let mut res = PertResult {
        name: name.clone(), n_cells: np, n_de: 0, n_target: 0, lfc_used: prm.lfc, ctrl_median: f32::NAN,
        frac_zero: f64::NAN, own_lfc: f64::NAN, mean_ps: f64::NAN, ps_sep: f64::NAN, fit: None, fitb: None, bins: vec![], status: "ok", cells: vec![],
    };
    if n_full < prm.min_cells || np < 20 {
        res.status = "few_cells";
        return res;
    }
    let own = gid.get(name.rsplit('|').next().unwrap()).copied();
    // dense block of perturbed cells
    let mut dense = vec![0f32; np * ng];
    let mut nzp = vec![0u32; ng];
    let mut linp = vec![0f64; ng];
    for i in 0..np {
        let s = c.scale[cells[i] as usize];
        let (ix, ct) = c.row(cells[i] as usize);
        let row = &mut dense[i * ng..(i + 1) * ng];
        for (&g, &k) in ix.iter().zip(ct) {
            let lin = k as f32 * s;
            row[g as usize] = lin.ln_1p();
            nzp[g as usize] += 1;
            linp[g as usize] += lin as f64;
        }
    }
    // 1. DE
    let n_all = (np + nc) as f64;
    let mut lfc = vec![0f64; ng];
    for g in 0..ng {
        let pct1 = nzp[g] as f64 / np as f64;
        let pct2 = (ctrl.off[g + 1] - ctrl.off[g]) as f64 / nc as f64;
        if pct1.max(pct2) < prm.min_pct {
            lfc[g] = f64::NAN;
            continue;
        }
        lfc[g] = (linp[g] / np as f64 + 1.0).log2() - (ctrl.sum_lin[g] / nc as f64 + 1.0).log2();
    }
    if let Some(o) = own {
        res.own_lfc = lfc[o];
    }
    let mut thr = prm.lfc;
    let mut de: Vec<usize>;
    let mut round = 0;
    loop {
        de = (0..ng).filter(|&g| lfc[g].abs() >= thr).collect();
        round += 1;
        if round >= 3 || de.len() >= prm.min_targets {
            break;
        }
        thr *= 0.8;
    }
    res.lfc_used = thr;
    res.n_de = de.len();
    if de.is_empty() {
        res.status = "no_de";
        return res;
    }
    // Wilcoxon z for DE genes (ties: zeros), continuity correction
    let (n1, n2) = (np as f64, nc as f64);
    let mut colbuf = vec![0f32; np];
    let mut zs: Vec<(usize, f64)> = de
        .iter()
        .map(|&g| {
            let cs = ctrl.nz(g);
            let cz = nc - cs.len();
            let mut u = 0f64;
            let mut pz = 0usize;
            for i in 0..np {
                colbuf[i] = dense[i * ng + g];
            }
            for &v in &colbuf {
                if v <= 0.0 {
                    pz += 1;
                    u += 0.5 * cz as f64;
                } else {
                    let lb = lower_bound(cs, v);
                    let ub = upper_bound(cs, v);
                    u += (cz + lb) as f64 + 0.5 * (ub - lb) as f64;
                }
            }
            // pert-pert ties among nonzero values contribute 0.5 each pair implicitly (ignored)
            let t = (pz + cz) as f64;
            let ntot = n1 + n2;
            let var = n1 * n2 / 12.0 * ((ntot + 1.0) - (t * t * t - t) / (ntot * (ntot - 1.0)));
            let d = u - n1 * n2 / 2.0;
            let z = if var > 0.0 { (d - 0.5 * d.signum()) / var.sqrt() } else { 0.0 };
            (g, z.abs())
        })
        .collect();
    zs.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    zs.truncate(prm.max_targets);
    // 2. target genes, detection >= 1% of P + control
    let tg: Vec<usize> = zs
        .iter()
        .map(|&(g, _)| g)
        .filter(|&g| prm.use_perturb_exp || Some(g) != own)
        .filter(|&g| (nzp[g] as f64 + (ctrl.off[g + 1] - ctrl.off[g]) as f64) >= 0.01 * n_all)
        .collect();
    res.n_target = tg.len();
    if tg.is_empty() {
        res.status = "no_target";
        return res;
    }
    // 3. clip + ridge
    let lam = prm.lambda;
    let det = (n_all + lam) * (n1 + lam) - n1 * n1;
    let mut q = vec![0f32; tg.len()];
    let mut bp = vec![0f64; tg.len()];
    let mut bn = vec![0f64; tg.len()];
    let mut pv: Vec<f32> = Vec::with_capacity(np);
    for (t, &g) in tg.iter().enumerate() {
        pv.clear();
        pv.extend((0..np).map(|i| dense[i * ng + g]).filter(|&v| v > 0.0));
        pv.sort_unstable_by(f32::total_cmp);
        let cs = ctrl.nz(g);
        let z = (np - pv.len()) + (nc - cs.len());
        let qt = q7_merged(z, &pv, cs, 0.95);
        q[t] = qt;
        let sp: f64 = pv.iter().map(|&v| v.min(qt) as f64).sum();
        let mut sc = ctrl.sum_log[g];
        for &v in cs.iter().rev() {
            if v <= qt {
                break;
            }
            sc -= (v - qt) as f64;
        }
        let sa = sp + sc;
        bn[t] = ((n1 + lam) * sa - n1 * sp) / det;
        bp[t] = ((n_all + lam) * sp - n1 * sa) / det;
    }
    let nb2: f64 = bp.iter().map(|v| v * v).sum();
    if nb2 <= 1e-300 {
        res.status = "zero_beta";
        return res;
    }
    let kk: f64 = bp.iter().zip(&bn).map(|(a, b)| a * b).sum();
    let sf = prm.scale_factor;
    // 4. per-cell scores
    let xp: Vec<f64> = (0..np)
        .map(|i| {
            let row = &dense[i * ng..(i + 1) * ng];
            let s: f64 = tg.iter().enumerate().map(|(t, &g)| bp[t] * row[g].min(q[t]) as f64).sum();
            ((s - kk) / nb2).clamp(0.0, sf) / sf
        })
        .collect();
    let mut acc = vec![0f64; nc];
    for (t, &g) in tg.iter().enumerate() {
        let (a, b) = (ctrl.off[g], ctrl.off[g + 1]);
        let (qt, bt) = (q[t], bp[t]);
        for k in a..b {
            acc[ctrl.cell[k] as usize] += bt * ctrl.val[k].min(qt) as f64;
        }
    }
    let mut mx = xp.iter().cloned().fold(0.0, f64::max);
    for &s in &acc {
        mx = mx.max(((s - kk) / nb2).clamp(0.0, sf) / sf);
    }
    if mx < 0.01 {
        mx = 1.0;
    }
    let ps: Vec<f64> = xp.iter().map(|v| v / mx).collect();
    res.mean_ps = ps.iter().sum::<f64>() / np as f64;
    {
        let (mut s1, mut s2) = (0f64, 0f64);
        for &s in &acc {
            let v = ((s - kk) / nb2).clamp(0.0, sf) / sf / mx;
            s1 += v;
            s2 += v * v;
        }
        let m = s1 / nc as f64;
        let sd = (s2 / nc as f64 - m * m).max(1e-12).sqrt();
        res.ps_sep = (res.mean_ps - m) / sd;
    }
    // 5. dose axis
    let Some(o) = own else {
        res.status = "gene_not_measured";
        return res;
    };
    // dose axis values for perturbed and control cells
    let lin_p: Vec<f32> = (0..np).map(|i| dense[i * ng + o].exp_m1()).collect();
    let mut lin_c = vec![0f32; nc];
    for k in ctrl.off[o]..ctrl.off[o + 1] {
        lin_c[ctrl.cell[k] as usize] = ctrl.val[k].exp_m1();
    }
    let (e, cv): (Vec<f32>, Vec<f32>) = match prm.axis {
        Axis::Log => (lin_p.iter().map(|v| v.ln_1p()).collect(), lin_c.iter().map(|v| v.ln_1p()).collect()),
        Axis::Lin => (lin_p.clone(), lin_c.clone()),
        Axis::Z => {
            let bt = c.batch.as_ref().expect("--axis z needs batch.bin in the cache");
            let nb = *bt.iter().max().unwrap() as usize + 1;
            let mut s1 = vec![0f64; nb];
            let mut s2 = vec![0f64; nb];
            let mut nn = vec![0f64; nb];
            for (i, &v) in lin_c.iter().enumerate() {
                let b = bt[ctrl.rows[i] as usize] as usize;
                s1[b] += v as f64;
                s2[b] += (v as f64) * (v as f64);
                nn[b] += 1.0;
            }
            let all_m = s1.iter().sum::<f64>() / nc as f64;
            let all_sd = ((s2.iter().sum::<f64>() / nc as f64 - all_m * all_m).max(0.0)).sqrt().max(1e-6);
            let ms: Vec<(f64, f64)> = (0..nb)
                .map(|b| {
                    if nn[b] < 2.0 {
                        return (all_m, all_sd);
                    }
                    let m = s1[b] / nn[b];
                    let sd = ((s2[b] / nn[b] - m * m).max(0.0)).sqrt();
                    (m, if sd > 1e-6 { sd } else { all_sd })
                })
                .collect();
            let z = |v: f32, row: u32| {
                let (m, sd) = ms[bt[row as usize] as usize];
                ((v as f64 - m) / sd) as f32
            };
            (
                lin_p.iter().enumerate().map(|(i, &v)| z(v, cells[i])).collect(),
                lin_c.iter().enumerate().map(|(i, &v)| z(v, ctrl.rows[i])).collect(),
            )
        }
    };
    let mut cs = cv.clone();
    cs.sort_unstable_by(f32::total_cmp);
    let med = {
        let h = (nc - 1) as f64 * 0.5;
        let lo = h.floor() as usize;
        let hi = (lo + 1).min(nc - 1);
        cs[lo] + (h - lo as f64) as f32 * (cs[hi] - cs[lo])
    };
    let mn = e.iter().cloned().fold(cs[0], f32::min);
    res.ctrl_median = med;
    res.frac_zero = lin_p.iter().filter(|&&v| v <= 0.0).count() as f64 / np as f64;
    let xs: Vec<f64> = if med > mn { e.iter().map(|&v| ((v - mn) / (med - mn)) as f64).collect() } else { vec![] };
    for i in 0..np {
        res.cells.push((cells[i], xs.get(i).copied().unwrap_or(f64::NAN) as f32, ps[i] as f32));
    }
    if xs.is_empty() {
        res.status = "ctrl_median_le_min";
        return res;
    }
    res.bins = make_bins(&xs, &ps, prm.bins, prm.bin_mode);
    let bx: Vec<f64> = res.bins.iter().map(|b| b.0).collect();
    let by: Vec<f64> = res.bins.iter().map(|b| b.1).collect();
    res.fit = hill::fit_ll4(&bx, &by, None);
    res.fitb = hill::fit_ll4(&bx, &by, Some((0.0, 1.0)));
    if res.fit.is_none() {
        res.status = "fit_failed";
    }
    res
}

fn q7(sorted: &[f64], p: f64) -> f64 {
    let h = (sorted.len() - 1) as f64 * p;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
}

pub fn make_bins(xs: &[f64], ps: &[f64], k: usize, mode: BinMode) -> Vec<(f64, f64, usize)> {
    let n = xs.len();
    let mut ord: Vec<usize> = (0..n).collect();
    ord.sort_by(|&a, &b| xs[a].total_cmp(&xs[b]).then(a.cmp(&b)));
    let mut lab = vec![0usize; n];
    let nb;
    match mode {
        BinMode::Rank => {
            for (r, &i) in ord.iter().enumerate() {
                lab[i] = r * k / n;
            }
            nb = k;
        }
        BinMode::CutUnique => {
            let sx: Vec<f64> = ord.iter().map(|&i| xs[i]).collect();
            let mut br: Vec<f64> = (0..=k).map(|j| q7(&sx, j as f64 / k as f64)).collect();
            br.dedup_by(|a, b| a == b);
            if br.len() < 2 {
                return vec![];
            }
            nb = br.len() - 1;
            for i in 0..n {
                // (br[j], br[j+1]], first bin closed on the left
                let v = xs[i];
                let j = br[1..].partition_point(|&b| b < v).min(nb - 1);
                lab[i] = j;
            }
        }
    }
    let mut out = vec![(0f64, 0f64, 0usize); nb];
    for i in 0..n {
        let b = &mut out[lab[i]];
        b.0 += xs[i];
        b.1 += ps[i];
        b.2 += 1;
    }
    out.into_iter().filter(|b| b.2 > 0).map(|(x, p, m)| (x / m as f64, p / m as f64, m)).collect()
}

pub fn run(dir: &str, out: &str, prm: &Params, only: Option<Vec<String>>, write_cells: bool) -> Result<()> {
    let t0 = Instant::now();
    let c = Cache::open(dir)?;
    let ng = c.genes.len();
    eprintln!("[ps] control '{}' {} cells; {} groups; {} genes", c.groups[0].0, c.groups[0].2, c.groups.len() - 1, ng);
    // control split: with --null, 25% of control cells are held out to form
    // size-matched pseudo-perturbations that go through the same pipeline
    let ctrl_all = c.cells(0);
    let (ref_c, hold): (Vec<u32>, Vec<u32>) = if prm.null > 0 {
        ctrl_all.iter().partition(|&&r| {
            let mut z = r as u64 ^ 0x5DEECE66D;
            splitmix(&mut z) % 4 != 0
        })
    } else {
        (ctrl_all.to_vec(), vec![])
    };
    let ctrl = Ctrl::build(&c, &ref_c, ng);
    eprintln!("[ps] control matrix ready {:.1}s (nnz {}, reference {}, held out {})", t0.elapsed().as_secs_f64(), ctrl.cell.len(), ref_c.len(), hold.len());
    let gid: HashMap<&str, usize> = c.genes.iter().enumerate().map(|(i, g)| (g.as_str(), i)).collect();
    let mut todo: Vec<usize> = (1..c.groups.len()).collect();
    if let Some(o) = &only {
        todo.retain(|&g| o.contains(&c.groups[g].0));
    }
    let mut jobs: Vec<(String, Vec<u32>)> = todo.iter().map(|&g| (c.groups[g].0.clone(), c.cells(g).to_vec())).collect();
    if prm.null > 0 {
        // each mock group borrows the size and the target gene of a real group,
        // so dose-axis methods can be run on it ("NULL_k|GENE")
        let src: Vec<usize> = (1..c.groups.len()).filter(|&g| c.groups[g].2 >= prm.min_cells).collect();
        let mut rng = 42u64;
        for k in 0..prm.null {
            let sg = src[(splitmix(&mut rng) % src.len() as u64) as usize];
            let n = c.groups[sg].2.min(hold.len());
            let mut pool = hold.clone();
            for i in 0..n {
                let j = i + (splitmix(&mut rng) % (pool.len() - i) as u64) as usize;
                pool.swap(i, j);
            }
            pool.truncate(n);
            jobs.push((format!("NULL_{k}|{}", c.groups[sg].0), pool));
        }
    }
    let t1 = Instant::now();
    let results: Vec<PertResult> = jobs.par_iter().map(|(n, cl)| one(&c, &ctrl, &gid, n, cl, prm)).collect();
    eprintln!("[ps] {} perturbations in {:.1}s", results.len(), t1.elapsed().as_secs_f64());

    let mut w = BufWriter::new(File::create(format!("{out}.genes.tsv"))?);
    writeln!(w, "gene\tn_cells\tn_de\tn_target\tlfc_used\town_log2fc\tctrl_median\tfrac_zero\tmean_ps\tps_sep\tn_bins\tb\tc\td\ted50\trss\thill0\thill1\tdelta\tkeep\tclass\tb_bd\tc_bd\td_bd\ted50_bd\trss_bd\tdelta_bd\tclass_bd\tstatus")?;
    let mut wb = BufWriter::new(File::create(format!("{out}.bins.tsv"))?);
    writeln!(wb, "gene\tbin\tmean_x\tmean_ps\tn")?;
    let mut wc = if write_cells { Some(BufWriter::new(File::create(format!("{out}.cells.tsv"))?)) } else { None };
    if let Some(w) = wc.as_mut() {
        writeln!(w, "gene\tcell_row\tx\tps")?;
    }
    let (mut nk, mut nbuf) = (0, 0);
    for r in &results {
        let (fs, h0, h1, keep, class) = match &r.fit {
            Some(f) => {
                let h0 = hill::predict(f, 0.0);
                let h1 = hill::predict(f, 1.0);
                let keep = h0 - h1 >= 0.2;
                let class = if !keep { "NA" } else if f.e <= 0.5 { "buffered" } else { "sensitive" };
                if keep {
                    nk += 1;
                    if f.e <= 0.5 {
                        nbuf += 1;
                    }
                }
                (format!("{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6e}", f.b, f.c, f.d, f.e, f.rss), h0, h1, keep, class)
            }
            None => ("NA\tNA\tNA\tNA\tNA".into(), f64::NAN, f64::NAN, false, "NA"),
        };
        let (fsb, db, classb) = match &r.fitb {
            Some(f) => {
                let d = hill::predict(f, 0.0) - hill::predict(f, 1.0);
                let cl = if d < 0.2 { "NA" } else if f.e <= 0.5 { "buffered" } else { "sensitive" };
                (format!("{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6e}", f.b, f.c, f.d, f.e, f.rss), d, cl)
            }
            None => ("NA\tNA\tNA\tNA\tNA".into(), f64::NAN, "NA"),
        };
        writeln!(
            w,
            "{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{}\t{}\t{}\t{:.4}\t{}\t{}",
            r.name, r.n_cells, r.n_de, r.n_target, r.lfc_used, r.own_lfc, r.ctrl_median, r.frac_zero, r.mean_ps, r.ps_sep,
            r.bins.len(), fs, h0, h1, h0 - h1, keep as u8, class, fsb, db, classb, r.status
        )?;
        for (j, b) in r.bins.iter().enumerate() {
            writeln!(wb, "{}\t{}\t{:.6}\t{:.6}\t{}", r.name, j + 1, b.0, b.1, b.2)?;
        }
        if let Some(w) = wc.as_mut() {
            for &(cell, x, p) in &r.cells {
                writeln!(w, "{}\t{}\t{:.5}\t{:.5}", r.name, cell, x, p)?;
            }
        }
    }
    eprintln!("[ps] kept {nk} fits, buffered {nbuf}, sensitive {}; total {:.1}s", nk - nbuf, t0.elapsed().as_secs_f64());
    Ok(())
}
