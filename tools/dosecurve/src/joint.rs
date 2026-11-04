//! Cross-fitted joint dose model, plus a Mixscale-style two-step baseline.
//! Specification: PLAN.md, section "Idea 1".
//!
//! Joint model, per perturbation:
//!   x_ig = (ln(1 + 1e4 n_ig / L_i) - mu_g,batch) / sd_g     (genes G, target excluded)
//!   x_ig = h(k_i) beta_g + e,   h(k) = k^gamma,   k_i ~ pi (grid, k = 0 escaper)
//!   n_it ~ NB(L_i a_t (1 - k_i), r_t)                        (target gene counts)
//!   beta_g ~ adaptive-shrinkage normal scale mixture (EB)
//! Cells are split in two folds; beta from one fold scores the other fold, and
//! the null variance of the score comes from NTC control cells.

use crate::hill;
use crate::ps::{self, splitmix, Cache};
use anyhow::Result;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

#[derive(Clone, Copy, PartialEq)]
pub enum Method {
    Joint,
    Mixscale,
}

pub struct JParams {
    pub method: Method,
    pub min_cells: usize,
    pub half: Option<u32>,
    pub null: usize,
    pub min_det: f64,
    pub n_ctrl_var: usize,
    pub outer: usize,
    pub write_cells: bool,
    /// diagnostic: ignore target counts (gamma should then be unidentified)
    pub no_target: bool,
    /// semi-synthetic: inject a known dose response into mock groups
    pub sim: Option<Sim>,
}

#[derive(Clone)]
pub struct Sim {
    pub gamma: f64,
    /// 'u': k ~ U(0.2, 0.98); 'c': k = 0.98 - 0.4 U^2 (concentrated near strong knockdown)
    pub kdist: char,
    pub eff: f64,
    pub n_genes: usize,
    pub escape: f64,
    /// draw target counts and scores exactly from the model (checks the estimator only)
    /// 'p' both, 'c' target counts only, 's' scores only, ' ' none
    pub pure: char,
    /// per cell response heterogeneity: x += k^gamma b s_i, log s_i ~ N(0, tau^2)
    pub tau: f64,
}

fn unif(st: &mut u64) -> f64 {
    (splitmix(st) >> 11) as f64 / (1u64 << 53) as f64
}

/// Inject x_ig += k_i^gamma beta_g into n_genes random genes and thin target
/// counts by (1 - k_i). Returns a description of the truth.
fn simulate(sp: &Sim, name: &str, d: &mut Dense, cnt: &mut [f64], drop_g: Option<usize>) -> Vec<f64> {
    let mut st = name.bytes().fold(0xC0FFEEu64, |h, b| h.wrapping_mul(131).wrapping_add(b as u64));
    let n = d.n;
    let ks: Vec<f64> = (0..n)
        .map(|_| {
            if unif(&mut st) < sp.escape {
                0.0
            } else if sp.kdist == 'u' {
                0.2 + 0.78 * unif(&mut st)
            } else {
                let u = unif(&mut st);
                0.98 - 0.4 * u * u
            }
        })
        .collect();
    let mut genes = vec![];
    while genes.len() < sp.n_genes.min(d.g - 1) {
        let g = (splitmix(&mut st) % d.g as u64) as usize;
        if Some(g) != drop_g && !genes.contains(&g) {
            genes.push(g);
        }
    }
    // x += h b  <=>  z += h b on the sparse rows (add explicit entries)
    let mut coef = vec![0f32; d.g];
    for (j, &g) in genes.iter().enumerate() {
        coef[g] = if j % 2 == 0 { sp.eff as f32 } else { -sp.eff as f32 };
    }
    let mut seen = vec![false; d.g];
    let (mut off, mut pos, mut z) = (vec![0u32], vec![], vec![]);
    for i in 0..n {
        let si = if sp.tau > 0.0 { (sp.tau * std_normal(&mut st) - 0.5 * sp.tau * sp.tau).exp() } else { 1.0 };
        let h = if ks[i] > 0.0 { (ks[i].powf(sp.gamma) * si) as f32 } else { 0.0 };
        for k in d.off[i] as usize..d.off[i + 1] as usize {
            let p = d.pos[k] as usize;
            pos.push(p as u16);
            z.push(d.z[k] + h * coef[p]);
            seen[p] = true;
        }
        for &g in &genes {
            if !seen[g] {
                pos.push(g as u16);
                z.push(h * coef[g]);
            }
        }
        for k in d.off[i] as usize..d.off[i + 1] as usize {
            seen[d.pos[k] as usize] = false;
        }
        off.push(pos.len() as u32);
    }
    d.off = off;
    d.pos = pos;
    d.z = z;
    for i in 0..n {
        let c = cnt[i] as u64;
        let mut kept = 0;
        for _ in 0..c {
            if unif(&mut st) >= ks[i] {
                kept += 1;
            }
        }
        cnt[i] = kept as f64;
    }
    ks
}

fn std_normal(st: &mut u64) -> f64 {
    let (a, b) = (unif(st).max(1e-300), unif(st));
    (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
}
/// Gamma(shape, 1), Marsaglia-Tsang
fn rgamma(st: &mut u64, shape: f64) -> f64 {
    if shape < 1.0 {
        return rgamma(st, shape + 1.0) * unif(st).max(1e-300).powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = std_normal(st);
        let v = (1.0 + c * x).powi(3);
        if v <= 0.0 {
            continue;
        }
        if unif(st).max(1e-300).ln() < 0.5 * x * x + d - d * v + d * v.ln() {
            return d * v;
        }
    }
}
fn rpois(st: &mut u64, mu: f64) -> f64 {
    if mu > 50.0 {
        return (mu + mu.sqrt() * std_normal(st)).round().max(0.0);
    }
    let l = (-mu).exp();
    let (mut k, mut p) = (0.0, 1.0);
    loop {
        p *= unif(st);
        if p <= l {
            return k;
        }
        k += 1.0;
    }
}

const NK: usize = 51;
fn kgrid() -> [f64; NK] {
    let mut k = [0f64; NK];
    for j in 0..NK {
        k[j] = (j as f64 / (NK - 1) as f64).min(0.995);
    }
    k
}
fn ggrid() -> Vec<f64> {
    // 0.25 .. 4, 17 values, log spaced
    (0..17).map(|i| (0.25f64.ln() + i as f64 * (16f64.ln() / 16.0)).exp()).collect()
}

#[inline]
fn hash(r: u32) -> u64 {
    let mut z = (r as u64).wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// upper tail of N(0,1)
fn norm_sf(z: f64) -> f64 {
    0.5 * erfc(z / std::f64::consts::SQRT_2)
}
fn norm_cdf(z: f64) -> f64 {
    1.0 - norm_sf(z)
}
fn erfc(x: f64) -> f64 {
    // Numerical Recipes erfcc, relative error < 1.2e-7
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let r = t * (-z * z - 1.26551223
        + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807
        + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))))).exp();
    if x >= 0.0 { r } else { 2.0 - r }
}

/// Global quantities from reference NTC controls.
struct Ref {
    gsel: Vec<usize>,       // global gene index of each modelled gene
    gpos: Vec<i32>,         // global -> modelled index, -1 if not modelled
    nb: usize,
    mu: Vec<f32>,           // nb x G, per batch control mean of y
    mz: Vec<f32>,           // nb x G, mu / sd
    sd: Vec<f32>,           // G
    // control subsample for score variance: sparse rows of y/sd on G
    cv_off: Vec<usize>,
    cv_idx: Vec<u32>,
    cv_val: Vec<f32>,
    cv_batch: Vec<u16>,
    cv_ll: Vec<f32>,        // log library size of the score set cells
    cv_rows: Vec<u32>,      // cache rows of the score set cells
    // per global gene sums for NB of the target: sum c, sum c^2, sum c L
    s_c: Vec<f64>,
    s_cc: Vec<f64>,
    s_cl: Vec<f64>,
    s_l: f64,
    s_ll: f64,
    // per global gene control values of 1e4 c / L (for the two-step dose axis median)
    n_ref: usize,
}

fn loglib(c: &Cache, r: u32) -> f64 {
    let s = c.scale[r as usize];
    if s > 0.0 { (1e4 / s as f64).ln() } else { 0.0 }
}

fn batch_of(c: &Cache, r: u32) -> usize {
    c.batch.as_ref().map_or(0, |b| b[r as usize] as usize)
}

fn build_ref(c: &Cache, refc_all: &[u32], prm: &JParams) -> Ref {
    let ng = c.genes.len();
    let nb = c.batch.as_ref().map_or(1, |b| *b.iter().max().unwrap() as usize + 1);
    // Split reference controls: a score set (calibrates the null of the cell
    // scores) that is NOT used for the gene means, so it is exchangeable with
    // perturbed cells, and a mean set for mu / sd / target NB parameters.
    let mut pool = refc_all.to_vec();
    let mut rng = 7u64;
    let m = prm.n_ctrl_var.min(pool.len() / 3);
    for i in 0..m {
        let j = i + (splitmix(&mut rng) % (pool.len() - i) as u64) as usize;
        pool.swap(i, j);
    }
    let refc: &[u32] = &pool[m..];
    let n = refc.len();
    // pass over reference controls, parallel over batches: each batch owns
    // its slice of the per batch sums, so no per thread copies are needed
    struct Acc {
        s1: Vec<f64>,
        s2: Vec<f64>,
        det: Vec<u32>,
        nnb: Vec<f64>,
        s_c: Vec<f64>,
        s_cc: Vec<f64>,
        s_cl: Vec<f64>,
        s_l: f64,
        s_ll: f64,
    }
    let mut by_b: Vec<Vec<u32>> = vec![vec![]; nb];
    for &r in refc {
        by_b[batch_of(c, r)].push(r);
    }
    let mut s1 = vec![0f64; nb * ng];
    let mut s2 = vec![0f64; nb * ng];
    let parts: Vec<(Vec<u32>, Vec<f64>, Vec<f64>, Vec<f64>, f64, f64)> = s1
        .par_chunks_mut(ng)
        .zip(s2.par_chunks_mut(ng))
        .zip(by_b.par_iter())
        .map(|((a1, a2), cells)| {
            let (mut det, mut sc, mut scc, mut scl) = (vec![0u32; ng], vec![0f64; ng], vec![0f64; ng], vec![0f64; ng]);
            let (mut sl, mut sll) = (0.0, 0.0);
            for &r in cells {
                let s = c.scale[r as usize];
                let l = if s > 0.0 { 1e4 / s as f64 } else { 0.0 };
                sl += l;
                sll += l * l;
                let (ix, ct) = c.row(r as usize);
                for (&g, &k) in ix.iter().zip(ct) {
                    let g = g as usize;
                    let y = (k as f32 * s).ln_1p() as f64;
                    a1[g] += y;
                    a2[g] += y * y;
                    det[g] += 1;
                    let kf = k as f64;
                    sc[g] += kf;
                    scc[g] += kf * kf;
                    scl[g] += kf * l;
                }
            }
            (det, sc, scc, scl, sl, sll)
        })
        .collect();
    let mut acc = Acc { s1, s2, det: vec![0; ng], nnb: by_b.iter().map(|v| v.len() as f64).collect(), s_c: vec![0.0; ng], s_cc: vec![0.0; ng], s_cl: vec![0.0; ng], s_l: 0.0, s_ll: 0.0 };
    for (det, sc, scc, scl, sl, sll) in parts {
        for g in 0..ng {
            acc.det[g] += det[g];
            acc.s_c[g] += sc[g];
            acc.s_cc[g] += scc[g];
            acc.s_cl[g] += scl[g];
        }
        acc.s_l += sl;
        acc.s_ll += sll;
    }
    // per batch means (fall back to the pooled mean for tiny batches), pooled within-batch sd
    let mut gsel = vec![];
    let mut gpos = vec![-1i32; ng];
    let mut sd_all = vec![0f64; ng];
    let mut mu_all = vec![0f64; nb * ng];
    for g in 0..ng {
        let tot: f64 = (0..nb).map(|b| acc.s1[b * ng + g]).sum();
        let pooled = tot / n as f64;
        let mut ss = 0.0;
        let mut used = 0.0;
        for b in 0..nb {
            let m = acc.nnb[b];
            if m >= 5.0 {
                let mb = acc.s1[b * ng + g] / m;
                mu_all[b * ng + g] = mb;
                ss += acc.s2[b * ng + g] - m * mb * mb;
                used += m - 1.0;
            } else {
                mu_all[b * ng + g] = pooled;
                ss += acc.s2[b * ng + g] - 2.0 * pooled * acc.s1[b * ng + g] + m * pooled * pooled;
                used += m;
            }
        }
        sd_all[g] = (ss.max(0.0) / used.max(1.0)).sqrt();
        if acc.det[g] as f64 >= prm.min_det * n as f64 && sd_all[g] > 1e-6 {
            gpos[g] = gsel.len() as i32;
            gsel.push(g);
        }
    }
    let ngs = gsel.len();
    let mut mu = vec![0f32; nb * ngs];
    let mut sd = vec![0f32; ngs];
    let mut mz = vec![0f32; nb * ngs];
    for (j, &g) in gsel.iter().enumerate() {
        sd[j] = sd_all[g] as f32;
        for b in 0..nb {
            mu[b * ngs + j] = mu_all[b * ng + g] as f32;
            mz[b * ngs + j] = (mu_all[b * ng + g] / sd_all[g]) as f32;
        }
    }
    // control score set
    pool.truncate(m);
    let (mut cv_off, mut cv_idx, mut cv_val, mut cv_batch, mut cv_ll) = (vec![0usize], vec![], vec![], vec![], vec![]);
    for &r in &pool {
        let s = c.scale[r as usize];
        let (ix, ct) = c.row(r as usize);
        for (&g, &k) in ix.iter().zip(ct) {
            let p = gpos[g as usize];
            if p >= 0 {
                cv_idx.push(p as u32);
                cv_val.push((k as f32 * s).ln_1p() / sd[p as usize]);
            }
        }
        cv_off.push(cv_idx.len());
        cv_batch.push(batch_of(c, r) as u16);
        cv_ll.push(loglib(c, r) as f32);
    }
    Ref { gsel, gpos, nb, mu, mz, sd, cv_off, cv_idx, cv_val, cv_batch, cv_ll, cv_rows: pool.clone(), s_c: acc.s_c, s_cc: acc.s_cc, s_cl: acc.s_cl, s_l: acc.s_l, s_ll: acc.s_ll, n_ref: n }
}

/// Null of a cell score from the control score set.
/// Scores depend on sequencing depth through imperfect normalisation; the
/// depth slope is estimated on control cells only (pooled within batch) and
/// removed from every score: u_adj = u - cmb[b] - slope (log L - mll[b]).
#[derive(Clone)]
struct CStat {
    cmb: Vec<f64>, // per batch centring value (control mean of the score)
    mll: Vec<f64>, // per batch control mean of log library size
    mb: Vec<f64>,  // control cells behind each centring value
    slope: f64,    // depth slope
    sw2: f64,      // pooled within-batch residual variance after the depth term
    sw2_raw: f64,  // same without the depth term
    // noise variance as a function of depth: v(d) = va + vb d, d = log L - mll[b]
    va: f64,
    vb: f64,
    // score set residuals (after centring and depth term), batch, log depth
    res: Vec<(f32, u16, f32, u32)>,
}

impl CStat {
    fn adj(&self, u: f64, b: usize, ll: f64) -> f64 {
        u - self.cmb[b] - self.slope * (ll - self.mll[b])
    }
    /// noise variance of one cell score, including the error of its batch centre
    fn var(&self, b: usize, ll: f64) -> f64 {
        let d = ll - self.mll[b];
        (self.va + self.vb * d).max(0.25 * self.va) * (1.0 + 1.0 / self.mb[b])
    }
}

/// Score the control score set with weights w (length G).
fn ctrl_score(rf: &Ref, w: &[f64]) -> CStat {
    let kb = wconst(rf, w);
    let m = rf.cv_batch.len();
    let nb = rf.nb;
    let (mut s1, mut s2, mut nn, mut l1, mut l2, mut ul) = (vec![0f64; nb], vec![0f64; nb], vec![0f64; nb], vec![0f64; nb], vec![0f64; nb], vec![0f64; nb]);
    let mut us = vec![0f64; m];
    for i in 0..m {
        let mut u = 0.0;
        for k in rf.cv_off[i]..rf.cv_off[i + 1] {
            u += w[rf.cv_idx[k] as usize] * rf.cv_val[k] as f64;
        }
        let b = rf.cv_batch[i] as usize;
        let l = rf.cv_ll[i] as f64;
        u -= kb[b];
        us[i] = u;
        s1[b] += u;
        s2[b] += u * u;
        nn[b] += 1.0;
        l1[b] += l;
        l2[b] += l * l;
        ul[b] += u * l;
    }
    let gm = s1.iter().sum::<f64>() / m as f64;
    let glm = l1.iter().sum::<f64>() / m as f64;
    let (mut suu, mut sll, mut sul, mut df) = (0.0, 0.0, 0.0, 0.0);
    let mut cmb = vec![gm; nb];
    let mut mll = vec![glm; nb];
    let mut mb = vec![m as f64; nb];
    for b in 0..nb {
        if nn[b] >= 5.0 {
            let mu = s1[b] / nn[b];
            let ml = l1[b] / nn[b];
            cmb[b] = mu;
            mll[b] = ml;
            mb[b] = nn[b];
            suu += s2[b] - nn[b] * mu * mu;
            sll += l2[b] - nn[b] * ml * ml;
            sul += ul[b] - nn[b] * mu * ml;
            df += nn[b] - 1.0;
        }
    }
    let slope = if sll > 0.0 { sul / sll } else { 0.0 };
    let res = suu - slope * sul;
    let sw2 = (res / (df - 1.0).max(1.0)).max(1e-12);
    // squared residual on centred depth (OLS) gives the variance function
    let (mut se, mut sed, mut sdd, mut sd, mut k) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 0..m {
        let b = rf.cv_batch[i] as usize;
        if nn[b] < 5.0 {
            continue;
        }
        let d = rf.cv_ll[i] as f64 - mll[b];
        let e = us[i] - cmb[b] - slope * d;
        let e2 = e * e * nn[b] / (nn[b] - 1.0);
        se += e2;
        sed += e2 * d;
        sdd += d * d;
        sd += d;
        k += 1.0;
    }
    let (va, vb) = if k > 10.0 && sdd > 0.0 {
        let md = sd / k;
        let vb = (sed - md * se) / (sdd - k * md * md);
        (se / k - vb * md, vb)
    } else {
        (sw2, 0.0)
    };
    let resv: Vec<(f32, u16, f32, u32)> = (0..m)
        .filter(|&i| nn[rf.cv_batch[i] as usize] >= 5.0)
        .map(|i| {
            let b = rf.cv_batch[i] as usize;
            let e = us[i] - cmb[b] - slope * (rf.cv_ll[i] as f64 - mll[b]);
            (e as f32, b as u16, rf.cv_ll[i], i as u32)
        })
        .collect();
    CStat { cmb, mll, mb, slope, sw2, sw2_raw: (suu / df.max(1.0)).max(1e-12), va, vb, res: resv }
}

/// Adaptive shrinkage (ash, normal scale mixture with a point mass at 0).
/// Returns posterior means and local false sign rates.
fn ash(bhat: &[f64], se: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = bhat.len();
    let semin = se.iter().cloned().filter(|&s| s > 0.0).fold(f64::INFINITY, f64::min);
    let smin = semin / 10.0;
    let mut smax2: f64 = 0.0;
    for i in 0..n {
        smax2 = smax2.max(bhat[i] * bhat[i] - se[i] * se[i]);
    }
    let smax = (2.0 * smax2.max(0.0).sqrt()).max(smin * 2.0);
    let mut grid = vec![0.0];
    let mut s = smin;
    while s < smax {
        grid.push(s);
        s *= std::f64::consts::SQRT_2;
    }
    grid.push(smax);
    let k = grid.len();
    // likelihood matrix (scaled per row for stability)
    let mut lik = vec![0f64; n * k];
    for i in 0..n {
        let mut mx = f64::NEG_INFINITY;
        for l in 0..k {
            let v = grid[l] * grid[l] + se[i] * se[i];
            let ll = -0.5 * (bhat[i] * bhat[i] / v + v.ln());
            lik[i * k + l] = ll;
            mx = mx.max(ll);
        }
        for l in 0..k {
            lik[i * k + l] = (lik[i * k + l] - mx).exp();
        }
    }
    let mut w = vec![1.0 / k as f64; k];
    w[0] = 0.5;
    let rest = 0.5 / (k - 1) as f64;
    for l in 1..k {
        w[l] = rest;
    }
    let nullpen = 9.0; // ash default nullweight 10
    let mut cnt = vec![0f64; k];
    for _ in 0..200 {
        cnt.iter_mut().for_each(|x| *x = 0.0);
        for i in 0..n {
            let row = &lik[i * k..(i + 1) * k];
            let mut d = 0.0;
            for l in 0..k {
                d += w[l] * row[l];
            }
            if d <= 0.0 {
                continue;
            }
            for l in 0..k {
                cnt[l] += w[l] * row[l] / d;
            }
        }
        cnt[0] += nullpen;
        let tot: f64 = cnt.iter().sum();
        let mut diff: f64 = 0.0;
        for l in 0..k {
            let nw = cnt[l] / tot;
            diff = diff.max((nw - w[l]).abs());
            w[l] = nw;
        }
        if diff < 1e-5 {
            break;
        }
    }
    let mut pm = vec![0f64; n];
    let mut lfsr = vec![1f64; n];
    for i in 0..n {
        let row = &lik[i * k..(i + 1) * k];
        let d: f64 = (0..k).map(|l| w[l] * row[l]).sum();
        if d <= 0.0 {
            continue;
        }
        let (mut m, mut pneg, mut ppos) = (0.0, 0.0, 0.0);
        for l in 0..k {
            let om = w[l] * row[l] / d;
            if l == 0 {
                pneg += om;
                ppos += om;
                continue;
            }
            let s2 = grid[l] * grid[l];
            let v = s2 + se[i] * se[i];
            let mu = bhat[i] * s2 / v;
            let sdp = (s2 * se[i] * se[i] / v).sqrt();
            m += om * mu;
            let pn = norm_cdf(-mu / sdp);
            pneg += om * pn;
            ppos += om * (1.0 - pn);
        }
        pm[i] = m;
        lfsr[i] = pneg.min(ppos);
    }
    (pm, lfsr)
}

/// Cells of one perturbation, sparse: z = y / sd on modelled genes.
/// The standardised value is x_ig = z_ig - mu_bg / sd_g (zero entries: z = 0),
/// so column sums and scores are computed without a dense n x G matrix.
struct Dense {
    n: usize,
    g: usize,
    off: Vec<u32>,
    pos: Vec<u16>,
    z: Vec<f32>,
    b: Vec<u16>,
    drop: Option<usize>,
}

fn dense(c: &Cache, rf: &Ref, cells: &[u32], drop_g: Option<usize>) -> Dense {
    let ng = rf.gsel.len();
    let n = cells.len();
    let (mut off, mut pos, mut z, mut bt) = (Vec::with_capacity(n + 1), vec![], vec![], Vec::with_capacity(n));
    off.push(0u32);
    for &r in cells {
        let s = c.scale[r as usize];
        let (ix, ct) = c.row(r as usize);
        for (&g, &k) in ix.iter().zip(ct) {
            let p = rf.gpos[g as usize];
            if p >= 0 && Some(p as usize) != drop_g {
                pos.push(p as u16);
                z.push((k as f32 * s).ln_1p() / rf.sd[p as usize]);
            }
        }
        off.push(pos.len() as u32);
        bt.push(batch_of(c, r) as u16);
    }
    Dense { n, g: ng, off, pos, z, b: bt, drop: drop_g }
}

/// Per gene sums over cells `sel`: (sum h x, sum x^2, sum h^2).
fn colsum(d: &Dense, rf: &Ref, sel: &[usize], h: &[f64]) -> (Vec<f64>, Vec<f64>, f64) {
    let g = d.g;
    let mut s1 = vec![0f64; g];
    let mut s2 = vec![0f64; g];
    let mut hb = vec![0f64; rf.nb];
    let mut nbc = vec![0f64; rf.nb];
    let mut hh = 0.0;
    for &i in sel {
        let hi = h[i];
        let b = d.b[i] as usize;
        hb[b] += hi;
        nbc[b] += 1.0;
        hh += hi * hi;
        let mzb = &rf.mz[b * g..(b + 1) * g];
        for k in d.off[i] as usize..d.off[i + 1] as usize {
            let p = d.pos[k] as usize;
            let zv = d.z[k] as f64;
            s1[p] += hi * zv;
            s2[p] += zv * (zv - 2.0 * mzb[p] as f64);
        }
    }
    for b in 0..rf.nb {
        if nbc[b] == 0.0 {
            continue;
        }
        let mzb = &rf.mz[b * g..(b + 1) * g];
        for p in 0..g {
            let m = mzb[p] as f64;
            s1[p] -= hb[b] * m;
            s2[p] += nbc[b] * m * m;
        }
    }
    if let Some(dg) = d.drop {
        s1[dg] = 0.0;
        s2[dg] = 0.0;
    }
    (s1, s2, hh)
}

/// beta (shrunk) from cells `sel` with weights h
fn fit_beta(d: &Dense, rf: &Ref, sel: &[usize], h: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let g = d.g;
    let (sxh, sxx, hh) = colsum(d, rf, sel, h);
    let m = sel.len() as f64;
    let mut bhat = vec![0f64; g];
    let mut se = vec![1f64; g];
    for j in 0..g {
        let b = sxh[j] / hh.max(1e-12);
        let rss = (sxx[j] - 2.0 * b * sxh[j] + b * b * hh).max(0.0);
        let s2 = (rss / (m - 1.0).max(1.0)).max(1e-4);
        bhat[j] = b;
        se[j] = (s2 / hh.max(1e-12)).sqrt();
    }
    if let Some(dg) = d.drop {
        bhat[dg] = 0.0;
        se[dg] = 1e6;
    }
    let (mut pm, lfsr) = ash(&bhat, &se);
    if let Some(dg) = d.drop {
        pm[dg] = 0.0;
    }
    (pm, lfsr, bhat)
}

/// per batch constant sum_g w_g mu_bg / sd_g
fn wconst(rf: &Ref, w: &[f64]) -> Vec<f64> {
    let g = rf.gsel.len();
    (0..rf.nb).map(|b| (0..g).map(|j| w[j] * rf.mz[b * g + j] as f64).sum()).collect()
}

/// score w . x_i (wc from wconst)
fn score(d: &Dense, i: usize, w: &[f64], wc: &[f64]) -> f64 {
    let mut u = 0.0;
    for k in d.off[i] as usize..d.off[i + 1] as usize {
        u += w[d.pos[k] as usize] * d.z[k] as f64;
    }
    u - wc[d.b[i] as usize]
}

pub struct JRes {
    name: String,
    n: usize,
    na: usize,
    nbb: usize,
    tmeas: bool,
    tmean: f64,
    t: f64,
    p: f64,
    n_de: usize,
    m: [f64; 2],
    v: [f64; 2],
    pi0: f64,
    mean_k: f64,
    gamma: f64,
    glo: f64,
    ghi: f64,
    ll_range: f64,
    rratio: f64,
    lr1: f64,
    p_lin: f64,
    tau2: f64,
    dose_mid: f64,
    rc: [f64; 4],
    status: &'static str,
    cells: Vec<(u32, f64, f64)>,
    // two-step
    ed50: f64,
    h0: f64,
    h1: f64,
    keep: bool,
    nbins: usize,
    boot: Option<Box<BootJob>>,
}

/// inputs of the linear-null bootstrap, run after all groups are fitted
struct BootJob {
    fold: Vec<usize>,
    cs: [CStat; 2],
    tc: Vec<f64>,
    lin: EmState,
    a: f64,
    r: f64,
    raw: f64,
}

impl JRes {
    fn empty(name: &str, n: usize, status: &'static str) -> JRes {
        JRes { name: name.into(), n, na: 0, nbb: 0, tmeas: false, tmean: f64::NAN, t: f64::NAN, p: f64::NAN, n_de: 0, m: [f64::NAN; 2], v: [f64::NAN; 2], pi0: f64::NAN, mean_k: f64::NAN, gamma: f64::NAN, glo: f64::NAN, ghi: f64::NAN, ll_range: f64::NAN, rratio: f64::NAN, lr1: f64::NAN, p_lin: f64::NAN, tau2: f64::NAN, dose_mid: f64::NAN, rc: [f64::NAN; 4], status, cells: vec![], ed50: f64::NAN, h0: f64::NAN, h1: f64::NAN, keep: false, nbins: 0, boot: None }
    }
}

const DP: usize = 4;

static PROF: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];
static EMIT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static EMCALL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
fn prof_add(k: usize, t: Instant) {
    PROF[k].fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
}

fn n_boot() -> usize {
    std::env::var("DOSE_BOOT").ok().and_then(|x| x.parse().ok()).unwrap_or(50)
}
fn tau_on() -> bool {
    std::env::var("DOSE_TAU").map_or(true, |x| x != "0")
}
fn robust_on() -> bool {
    std::env::var("DOSE_ROBUST").map_or(false, |x| x == "1")
}
thread_local! {
    /// anchored dose prior: (gamma, log-polynomial coefficients of the
    /// knockdown density fitted on target counts alone)
    static ANCH: std::cell::Cell<Option<(f64, [f64; DP])>> = const { std::cell::Cell::new(None) };
}
fn anchor_on() -> bool {
    std::env::var("DOSE_ANCHOR").map_or(false, |x| x == "1")
}
/// prior on the response grid implied by a fixed knockdown density p_k
/// (escaper mass p0 at h = 0): mass of h bin j = P(k^gamma in bin j)
fn anchored_pi(p0: f64, gamma: f64, thk: &[f64; DP]) -> [f64; NK] {
    const M: usize = 4000;
    let mut cdf = vec![0f64; M + 1];
    let mut acc = 0.0;
    let mut mx = f64::NEG_INFINITY;
    let e: Vec<f64> = (0..M).map(|i| {
        let k = (i as f64 + 0.5) / M as f64;
        let t = 2.0 * k - 1.0;
        let v = thk[0] * t + thk[1] * t * t + thk[2] * t * t * t + thk[3] * t * t * t * t;
        mx = mx.max(v);
        v
    }).collect();
    for i in 0..M {
        acc += (e[i] - mx).exp();
        cdf[i + 1] = acc;
    }
    let fk = |k: f64| -> f64 {
        let x = (k.clamp(0.0, 1.0) * M as f64).min(M as f64);
        let i = (x.floor() as usize).min(M - 1);
        let fr = x - i as f64;
        (cdf[i] + fr * (cdf[i + 1] - cdf[i])) / acc
    };
    let hg = hgrid();
    let dh = 1.0 / (NK - 1) as f64;
    let mut pi = [0f64; NK];
    let mut s = 0.0;
    for j in 1..NK {
        let lo = (hg[j] - 0.5 * dh).max(0.0);
        let hi = if j == NK - 1 { 1.0 } else { hg[j] + 0.5 * dh };
        pi[j] = (fk(hi.powf(1.0 / gamma)) - fk(lo.powf(1.0 / gamma))).max(1e-12);
        s += pi[j];
    }
    for j in 1..NK {
        pi[j] *= (1.0 - p0) / s;
    }
    pi[0] = p0;
    pi
}
fn with_anchor<T>(a: Option<(f64, [f64; DP])>, f: impl FnOnce() -> T) -> T {
    let old = ANCH.with(|c| c.replace(a));
    let r = f();
    ANCH.with(|c| c.set(old));
    r
}

thread_local! {
    static TOL_OVR: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
}
fn em_tol() -> f64 {
    let o = TOL_OVR.with(|c| c.get());
    if o > 0.0 {
        return o;
    }
    std::env::var("DOSE_TOL").ok().and_then(|x| x.parse().ok()).unwrap_or(1e-4)
}
/// convergence tolerance of the EM fits inside the bootstrap (only the
/// comparison of the replicate LR with the observed LR is needed)
fn boot_tol() -> f64 {
    std::env::var("DOSE_BOOT_TOL").ok().and_then(|x| x.parse().ok()).unwrap_or(1e-3)
}
fn em_tol_at(u: &[f64], fold: &[usize], vc: &[f64], nbl: &[f64], init: Option<EmState>, iters: usize, tol: f64) -> (f64, [f64; NK], Vec<f64>, EmState) {
    let old = TOL_OVR.with(|c| c.replace(tol));
    let r = em(u, fold, vc, nbl, init, iters);
    TOL_OVR.with(|c| c.set(old));
    r
}
fn em_iters() -> usize {
    std::env::var("DOSE_ITERS").ok().and_then(|x| x.parse().ok()).unwrap_or(300)
}


/// EM state carried between neighbouring gamma values (warm start)
#[derive(Clone, Copy)]
struct EmState {
    p0: f64,
    th: [f64; DP],
    m: [f64; 2],
    /// tau^2: dose-independent response heterogeneity, Var(u | h) = v + tau^2 m^2 h^2
    t: f64,
}

/// Response grid h_j = j / (NK - 1). The latent variable is the response
/// h_i = k_i^gamma itself, with the prior on h. The score likelihood
/// N(u_i; m h_i, v) therefore does not depend on gamma at all, and gamma
/// enters only through the target counts at k_j = h_j^(1/gamma). Without
/// target information the gamma profile is exactly flat, so the shape of
/// the score distribution can no longer masquerade as dose curvature.
fn hgrid() -> [f64; NK] {
    let mut h = [0f64; NK];
    for j in 0..NK {
        h[j] = j as f64 / (NK - 1) as f64;
    }
    h
}
fn kgrid_g(gamma: f64) -> [f64; NK] {
    let hg = hgrid();
    let mut k = [0f64; NK];
    for j in 0..NK {
        k[j] = if hg[j] <= 0.0 { 0.0 } else { hg[j].powf(1.0 / gamma).min(0.995) };
    }
    k
}

/// one EM fit for fixed gamma (nbl = NB log-lik table for this gamma);
/// returns (loglik, pi, posterior weights n x NK, state)
fn em(u: &[f64], fold: &[usize], vc: &[f64], nbl: &[f64], init: Option<EmState>, iters: usize) -> (f64, [f64; NK], Vec<f64>, EmState) {
    let n = u.len();
    let hk = hgrid();
    // prior on h: escaper mass at 0 + smooth log-polynomial density (degree 4)
    let phi: Vec<[f64; DP]> = hk.iter().map(|&k| {
        let t = 2.0 * k - 1.0;
        [t, t * t, t * t * t, t * t * t * t]
    }).collect();
    let poly_pi = |p0: f64, th: &[f64; DP]| -> [f64; NK] {
        let mut pi = [0f64; NK];
        let mut mx = f64::NEG_INFINITY;
        let mut e = [0f64; NK];
        for j in 1..NK {
            e[j] = (0..DP).map(|d| th[d] * phi[j][d]).sum::<f64>();
            mx = mx.max(e[j]);
        }
        let mut s = 0.0;
        for j in 1..NK {
            pi[j] = (e[j] - mx).exp();
            s += pi[j];
        }
        for j in 1..NK {
            pi[j] *= (1.0 - p0) / s;
        }
        pi[0] = p0;
        pi
    };
    let mut st = init.unwrap_or_else(|| {
        let mut m = [0f64; 2];
        for f in 0..2 {
            let (mut s, mut c): (f64, f64) = (0.0, 0.0);
            for i in 0..n {
                if fold[i] == f {
                    s += u[i];
                    c += 1.0;
                }
            }
            m[f] = (2.0 * s / c.max(1.0)).max(1e-3);
        }
        EmState { p0: 0.1, th: [0.0; DP], m, t: if tau_on() { 0.05 } else { 0.0 } }
    });
    EMCALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tau = tau_on();
    let anch = ANCH.with(|c| c.get());
    let prior = |p0: f64, th: &[f64; DP]| -> [f64; NK] { match anch { Some((g, thk)) => anchored_pi(p0, g, &thk), None => poly_pi(p0, th) } };
    let mut w = vec![0f64; n * NK];
    // one generalised EM step: E step at `st` (fills w, returns loglik at st),
    // then M step updating st in place
    let mut step = |st: &mut EmState, w: &mut [f64]| -> f64 {
        let pi = prior(st.p0, &st.th);
        let mut lpi = [0f64; NK];
        for j in 0..NK {
            lpi[j] = pi[j].max(1e-300).ln();
        }
        let mut lln = 0.0;
        for i in 0..n {
            let f = fold[i];
            let row = &mut w[i * NK..(i + 1) * NK];
            let nb = &nbl[i * NK..(i + 1) * NK];
            let mut mx = f64::NEG_INFINITY;
            let tm = st.t * st.m[f] * st.m[f];
            let ivc = 1.0 / vc[i];
            for j in 0..NK {
                let r = u[i] - st.m[f] * hk[j];
                let q = tm * hk[j] * hk[j];
                let sij = vc[i] + q;
                let l = lpi[j] + nb[j] - 0.5 * (q * ivc).ln_1p() - 0.5 * r * r / sij;
                row[j] = l;
                mx = mx.max(l);
            }
            let mut s = 0.0;
            for j in 0..NK {
                row[j] = (row[j] - mx).exp();
                s += row[j];
            }
            let is = 1.0 / s;
            for j in 0..NK {
                row[j] *= is;
            }
            lln += mx + s.ln();
        }
        // M step
        let mut np = [1e-3f64; NK];
        let mut num = [0f64; 2];
        let mut den = [0f64; 2];
        for i in 0..n {
            let f = fold[i];
            let tm = st.t * st.m[f] * st.m[f];
            let (mut a, mut b) = (0.0, 0.0);
            for j in 0..NK {
                let wij = w[i * NK + j];
                np[j] += wij;
                // weighted least squares for m, weights 1 / Var(u | h_j)
                let is = 1.0 / (vc[i] + tm * hk[j] * hk[j]);
                a += wij * hk[j] * is;
                b += wij * hk[j] * hk[j] * is;
            }
            num[f] += a * u[i];
            den[f] += b;
        }
        let tot: f64 = np.iter().sum();
        st.p0 = (np[0] / tot).clamp(1e-4, 0.9);
        let wk: f64 = np[1..].iter().sum();
        for _ in 0..(if anch.is_some() { 0 } else { 3 }) {
            let q = poly_pi(0.0, &st.th);
            let mut ef = [0f64; DP];
            for j in 1..NK {
                for d in 0..DP {
                    ef[d] += q[j] * phi[j][d];
                }
            }
            let mut g = [0f64; DP];
            let mut hm = [[0f64; DP]; DP];
            for j in 1..NK {
                for d in 0..DP {
                    g[d] += np[j] * phi[j][d] - wk * q[j] * phi[j][d];
                    for e in 0..DP {
                        hm[d][e] += wk * q[j] * (phi[j][d] - ef[d]) * (phi[j][e] - ef[e]);
                    }
                }
            }
            for d in 0..DP {
                hm[d][d] += 1e-3 * wk + 1e-6;
            }
            if let Some(stp) = solve_spd(&hm, &g) {
                for d in 0..DP {
                    st.th[d] = (st.th[d] + stp[d]).clamp(-60.0, 60.0);
                }
            }
        }
        for f in 0..2 {
            st.m[f] = (num[f] / den[f].max(1e-12)).max(0.0);
        }
        if tau {
            // guarded Newton step for log tau^2 on the expected complete
            // log-lik, analytic derivatives in one pass
            let t = st.t.max(1e-6);
            let (mut q0, mut d1, mut d2) = (0.0, 0.0, 0.0);
            for i in 0..n {
                let f = fold[i];
                let m2 = st.m[f] * st.m[f];
                for j in 1..NK {
                    let wij = w[i * NK + j];
                    if wij < 1e-8 {
                        continue;
                    }
                    let r = u[i] - st.m[f] * hk[j];
                    let b = m2 * hk[j] * hk[j];
                    let sij = vc[i] + t * b;
                    let (is, r2) = (1.0 / sij, r * r);
                    q0 -= wij * (0.5 * sij.ln() + 0.5 * r2 * is);
                    d1 += wij * (-0.5 * b * is + 0.5 * r2 * b * is * is);
                    d2 += wij * (0.5 * b * b * is * is - r2 * b * b * is * is * is);
                }
            }
            let g = t * d1;
            let h = t * d1 + t * t * d2;
            let stp = if h < 0.0 { (-g / h).clamp(-1.0, 1.0) } else { g.signum() * 0.5 };
            let lt = t.ln();
            let nt = (lt + stp).clamp((1e-6f64).ln(), 4f64.ln()).exp();
            let mut q1 = 0.0;
            for i in 0..n {
                let f = fold[i];
                let m2 = st.m[f] * st.m[f];
                for j in 1..NK {
                    let wij = w[i * NK + j];
                    if wij < 1e-8 {
                        continue;
                    }
                    let r = u[i] - st.m[f] * hk[j];
                    let sij = vc[i] + nt * m2 * hk[j] * hk[j];
                    q1 -= wij * (0.5 * sij.ln() + 0.5 * r * r / sij);
                }
            }
            if q1 > q0 {
                st.t = nt;
            }
        }
        lln
    };
    // SQUAREM acceleration (Varadhan and Roland 2008) on the transformed
    // parameters, with a monotone safeguard: an extrapolated point is kept only
    // if its loglik is not below that of the last plain EM step.
    let enc = |s: &EmState| -> [f64; 8] {
        let p = s.p0.clamp(1e-6, 1.0 - 1e-6);
        [(p / (1.0 - p)).ln(), s.th[0], s.th[1], s.th[2], s.th[3], s.m[0], s.m[1], s.t.max(1e-9).ln()]
    };
    let dec = |x: &[f64; 8], t_on: bool| -> EmState {
        EmState {
            p0: (1.0 / (1.0 + (-x[0]).exp())).clamp(1e-4, 0.9),
            th: [x[1].clamp(-60.0, 60.0), x[2].clamp(-60.0, 60.0), x[3].clamp(-60.0, 60.0), x[4].clamp(-60.0, 60.0)],
            m: [x[5].max(0.0), x[6].max(0.0)],
            t: if t_on { x[7].exp().clamp(1e-6, 4.0) } else { 0.0 },
        }
    };
    let accel = std::env::var("DOSE_SQUAREM").map_or(true, |x| x != "0");
    let mut ll = f64::NEG_INFINITY;
    let mut used = 0usize;
    let t_on = st.t > 0.0 || tau;
    while used < iters {
        if !accel || used + 3 > iters {
            let lln = step(&mut st, &mut w);
            used += 1;
            EMIT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let done = (lln - ll).abs() < em_tol();
            ll = lln;
            if done {
                break;
            }
            continue;
        }
        let x0 = enc(&st);
        let l0 = step(&mut st, &mut w);
        if (l0 - ll).abs() < em_tol() {
            ll = l0;
            used += 1;
            break;
        }
        let x1 = enc(&st);
        let s1 = st;
        let l1 = step(&mut st, &mut w);
        let x2 = enc(&st);
        let s2 = st;
        used += 2;
        EMIT.fetch_add(2, std::sync::atomic::Ordering::Relaxed);
        if (l1 - l0).abs() < em_tol() {
            ll = l1;
            break;
        }
        let mut rr = 0.0;
        let mut vv = 0.0;
        let mut r = [0f64; 8];
        let mut v = [0f64; 8];
        for k in 0..8 {
            r[k] = x1[k] - x0[k];
            v[k] = x2[k] - 2.0 * x1[k] + x0[k];
            rr += r[k] * r[k];
            vv += v[k] * v[k];
        }
        let alpha = if vv > 0.0 { -(rr / vv).sqrt() } else { -1.0 };
        if alpha >= -1.0 {
            // no extrapolation gain: continue from the second plain step
            ll = l1;
            continue;
        }
        let alpha = alpha.max(-50.0);
        let mut xn = [0f64; 8];
        for k in 0..8 {
            xn[k] = x0[k] - 2.0 * alpha * r[k] + alpha * alpha * v[k];
        }
        let mut sn = dec(&xn, t_on);
        let ln = step(&mut sn, &mut w);
        used += 1;
        EMIT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if ln.is_finite() && ln >= l1 {
            st = sn;
            let done = (ln - l1).abs() < em_tol();
            ll = ln;
            if done {
                break;
            }
        } else {
            // reject: back to the plain EM path (w is refreshed by the next step)
            let _ = s1;
            st = s2;
            ll = l1;
        }
    }
    // final E step at the returned state so that w, pi and ll match st
    let mut fin = st;
    let lf = step(&mut fin, &mut w);
    let _ = ll;
    let pi = prior(st.p0, &st.th);
    let ll = lf;
    let st = fin;
    (ll, pi, w, st)
}

/// prior on the response grid (same form as inside `em`)
fn prior_pi(p0: f64, th: &[f64; DP]) -> [f64; NK] {
    let hg = hgrid();
    let mut e = [0f64; NK];
    let mut mx = f64::NEG_INFINITY;
    for j in 1..NK {
        let t = 2.0 * hg[j] - 1.0;
        let f = [t, t * t, t * t * t, t * t * t * t];
        e[j] = (0..DP).map(|d| th[d] * f[d]).sum::<f64>();
        mx = mx.max(e[j]);
    }
    let mut pi = [0f64; NK];
    let mut s = 0.0;
    for j in 1..NK {
        pi[j] = (e[j] - mx).exp();
        s += pi[j];
    }
    for j in 1..NK {
        pi[j] *= (1.0 - p0) / s;
    }
    pi[0] = p0;
    pi
}

/// Inputs of the per cell marginal likelihood
struct LikIn<'a> {
    u: &'a [f64],
    fold: &'a [usize],
    vc: &'a [f64],
    cnt: &'a [f64],
    lib: &'a [f64],
    a: f64,
    r: f64,
}

/// per cell log marginal likelihood; th = [log gamma, m0, m1, logit p0, theta1..4]
fn cell_ll(x: &LikIn, th: &[f64; 8]) -> Vec<f64> {
    let g = th[0].exp();
    let nbl = nb_table(x.cnt, x.lib, x.a, x.r, g);
    let p0 = 1.0 / (1.0 + (-th[3]).exp());
    let pi = prior_pi(p0, &[th[4], th[5], th[6], th[7]]);
    let hk = hgrid();
    let mut lpi = [0f64; NK];
    for j in 0..NK {
        lpi[j] = pi[j].max(1e-300).ln();
    }
    let n = x.u.len();
    let mut out = vec![0f64; n];
    let mut l = [0f64; NK];
    for i in 0..n {
        let m = th[1 + x.fold[i]];
        let mut mx = f64::NEG_INFINITY;
        for j in 0..NK {
            let r = x.u[i] - m * hk[j];
            l[j] = lpi[j] + nbl[i * NK + j] - 0.5 * r * r / x.vc[i];
            mx = mx.max(l[j]);
        }
        out[i] = mx + l.iter().map(|&v| (v - mx).exp()).sum::<f64>().ln();
    }
    out
}

/// Ratio V_model / V_robust for log gamma at the fit (sandwich, cell level).
/// The profile likelihood ratio is scaled by min(1, ratio): a robust LR that
/// widens the gamma interval when the score noise departs from the model.
/// This is the analytic form of a one-step score bootstrap over cells: no
/// refitting of the gene weights or of the EM is needed.
fn robust_ratio(x: &LikIn, th: &[f64; 8]) -> f64 {
    // nuisance at a boundary is held fixed
    let act: Vec<usize> = (0..8).filter(|&k| !(k == 3 && (th[3] < -9.0 || th[3] > 2.1)) && !((k == 1 || k == 2) && th[k] <= 0.0)).collect();
    let p = act.len();
    let dl: Vec<f64> = (0..8).map(|k| 1e-3 * th[k].abs().max(if k == 1 || k == 2 { 1e-2 } else { 1.0 })).collect();
    let n = x.u.len();
    let at = |sh: &[(usize, f64)]| -> Vec<f64> {
        let mut t = *th;
        for &(k, d) in sh {
            t[k] += d;
        }
        cell_ll(x, &t)
    };
    let l0: f64 = at(&[]).iter().sum();
    // per cell gradients (central differences)
    let mut gp: Vec<Vec<f64>> = vec![];
    let mut lp = vec![0f64; p];
    let mut lm = vec![0f64; p];
    for (a, &k) in act.iter().enumerate() {
        let up = at(&[(k, dl[k])]);
        let dn = at(&[(k, -dl[k])]);
        lp[a] = up.iter().sum();
        lm[a] = dn.iter().sum();
        gp.push((0..n).map(|i| (up[i] - dn[i]) / (2.0 * dl[k])).collect());
    }
    // observed information I = -Hessian of the total
    let mut info = vec![vec![0f64; p]; p];
    for a in 0..p {
        let k = act[a];
        info[a][a] = -(lp[a] - 2.0 * l0 + lm[a]) / (dl[k] * dl[k]);
        for b in 0..a {
            let l = act[b];
            let pp: f64 = at(&[(k, dl[k]), (l, dl[l])]).iter().sum();
            let pm: f64 = at(&[(k, dl[k]), (l, -dl[l])]).iter().sum();
            let mp: f64 = at(&[(k, -dl[k]), (l, dl[l])]).iter().sum();
            let mm: f64 = at(&[(k, -dl[k]), (l, -dl[l])]).iter().sum();
            let h = (pp - pm - mp + mm) / (4.0 * dl[k] * dl[l]);
            info[a][b] = -h;
            info[b][a] = -h;
        }
    }
    // J = centred outer product of per cell gradients
    let mean: Vec<f64> = gp.iter().map(|g| g.iter().sum::<f64>() / n as f64).collect();
    let mut jm = vec![vec![0f64; p]; p];
    for i in 0..n {
        for a in 0..p {
            let ga = gp[a][i] - mean[a];
            for b in 0..=a {
                jm[a][b] += ga * (gp[b][i] - mean[b]);
            }
        }
    }
    for a in 0..p {
        for b in 0..a {
            jm[b][a] = jm[a][b];
        }
    }
    // c = I^{-1} e_gamma (gamma is act[0] = 0)
    let Some(c) = solve_dyn(&info, 0) else { return 1.0 };
    let vm = c[0];
    let mut vr = 0.0;
    for a in 0..p {
        for b in 0..p {
            vr += c[a] * jm[a][b] * c[b];
        }
    }
    if !(vm > 0.0 && vr > 0.0) {
        return 1.0;
    }
    vm / vr
}

/// Solve A x = e_k for a small symmetric positive definite A (Cholesky)
fn solve_dyn(a: &[Vec<f64>], k: usize) -> Option<Vec<f64>> {
    let p = a.len();
    let mut l = vec![vec![0f64; p]; p];
    for i in 0..p {
        for j in 0..=i {
            let mut s = a[i][j];
            for q in 0..j {
                s -= l[i][q] * l[j][q];
            }
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                l[i][i] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    let mut y = vec![0f64; p];
    for i in 0..p {
        let mut s = if i == k { 1.0 } else { 0.0 };
        for q in 0..i {
            s -= l[i][q] * y[q];
        }
        y[i] = s / l[i][i];
    }
    let mut x = vec![0f64; p];
    for i in (0..p).rev() {
        let mut s = y[i];
        for q in i + 1..p {
            s -= l[q][i] * x[q];
        }
        x[i] = s / l[i][i];
    }
    Some(x)
}

/// Parametric bootstrap p value of the linear model (gamma = 1).
/// Null data sets keep the fitted linear model (prior on h, m per fold,
/// target NB) but take the score noise and the depth of each cell from a
/// random NTC control of the score set, so non-Gaussian, depth dependent
/// noise is reproduced. Only the one-dimensional EM is refitted: the gene
/// weights stay fixed. Coarse gamma grid (every second value), warm started.
fn boot_linear(fold: &[usize], cs: [&CStat; 2], tc: &[f64], lin: EmState, a: f64, r: f64, lr_obs: f64, nb: usize, name: &str) -> f64 {
    let n = fold.len();
    let gg = ggrid();
    let mid = gg.len() / 2;
    let pi = prior_pi(lin.p0, &lin.th);
    let hg = hgrid();
    let mut cdf = [0f64; NK];
    let mut acc = 0.0;
    for j in 0..NK {
        acc += pi[j];
        cdf[j] = acc;
    }
    let mut up: Vec<usize> = (mid..gg.len()).step_by(2).collect();
    let down: Vec<usize> = (0..mid).rev().step_by(2).collect();
    let seed0 = name.bytes().fold(0xB007u64, |h, b| h.wrapping_mul(131).wrapping_add(b as u64));
    let seed0 = seed0 ^ std::env::var("DOSE_SEED").ok().and_then(|x| x.parse::<u64>().ok()).unwrap_or(0).wrapping_mul(0xD1B54A32D192ED03);
    // replicates run in parallel, each with its own random stream. The
    // sequential rule stopped at the first k exceedances with (1 + k) / (nb + 1)
    // >= 0.05; the same value is returned here.
    let kstop = (0..=nb).find(|&k| (1 + k) as f64 / (nb + 1) as f64 >= 0.05).unwrap_or(nb);
    let one = |bi: usize| -> usize {
        let mut st = seed0 ^ (bi as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15);
        let (mut u, mut vc, mut cnt, mut lib) = (vec![0f64; n], vec![0f64; n], vec![0f64; n], vec![0f64; n]);
        for i in 0..n {
            let f = fold[i];
            let x = unif(&mut st) * acc;
            let j = cdf.iter().position(|&c| c >= x).unwrap_or(NK - 1);
            let h = hg[j];
            let c = &cs[f].res[(splitmix(&mut st) % cs[f].res.len() as u64) as usize];
            u[i] = lin.m[f] * h * (1.0 + lin.t.sqrt() * std_normal(&mut st)) + c.0 as f64;
            vc[i] = cs[f].var(c.1 as usize, c.2 as f64);
            lib[i] = (c.2 as f64).exp();
            // the same control cell's target count, thinned by the knockdown
            let k = h.min(0.995);
            let c0 = tc[c.3 as usize];
            cnt[i] = if c0 > 200.0 {
                let p = 1.0 - k;
                (c0 * p + (c0 * p * k).sqrt() * std_normal(&mut st)).round().clamp(0.0, c0)
            } else {
                (0..c0 as u64).filter(|_| unif(&mut st) >= k).count() as f64
            };
        }
        let thb = if anchor_on() {
            let nbl = nb_table(&cnt, &lib, a, r, 1.0);
            let (_, _, _, sk) = with_anchor(None, || em(&vec![0f64; n], fold, &vec![1f64; n], &nbl, Some(EmState { p0: 0.1, th: [0.0; DP], m: [0.0; 2], t: 0.0 }), em_iters()));
            Some(sk.th)
        } else {
            None
        };
        let mut best = f64::NEG_INFINITY;
        let mut l1 = 0.0;
        let mut st_mid = None;
        for path in [&up, &down] {
            let mut prev = if st_mid.is_some() { st_mid } else { Some(lin) };
            for &gi in path.iter() {
                let nbl = nb_table(&cnt, &lib, a, r, gg[gi]);
                let (l, _, _, s2) = with_anchor(thb.map(|t| (gg[gi], t)), || em_tol_at(&u, fold, &vc, &nbl, prev, em_iters(), boot_tol()));
                if gi == mid {
                    l1 = l;
                    st_mid = Some(s2);
                }
                prev = Some(s2);
                best = best.max(l);
                // gamma = 1 is fitted first: stop once the replicate exceeds
                if st_mid.is_some() && 2.0 * (best - l1) >= lr_obs {
                    return 1;
                }
            }
        }
        0
    };
    // chunks of replicates in parallel, stopping as the sequential rule would
    let chunk = 5;
    let mut exceed = 0usize;
    let mut b0 = 0;
    while b0 < nb && exceed < kstop {
        let b1 = (b0 + chunk).min(nb);
        exceed += (b0..b1).into_par_iter().map(|bi| one(bi)).sum::<usize>();
        b0 = b1;
    }
    let _ = &mut up;
    let exceed = exceed.min(kstop);
    (1 + exceed) as f64 / (nb + 1) as f64
}

/// Regression calibration estimate of gamma (robust to the shape of
/// dose-independent response heterogeneity and of the score noise).
/// Model only for the mean: E[u_i | target count] = m_f E[k^gamma | count],
/// with the posterior of k from the target counts alone (wpost). Any cell
/// heterogeneity s_i with mean 1, independent of k, drops out of the mean.
/// gamma by weighted least squares on a fine grid; 95% interval from a cell
/// bootstrap (resampling cells within folds; posterior weights held fixed).
/// Returns [gamma, lo, hi, bootstrap two-sided p for gamma = 1].
fn regcal(u: &[f64], fold: &[usize], vc: &[f64], wpost: &[f64], name: &str) -> [f64; 4] {
    let n = u.len();
    let ng = 41;
    let gs: Vec<f64> = (0..ng).map(|i| (0.25f64.ln() + i as f64 * 16f64.ln() / (ng - 1) as f64).exp()).collect();
    let kg = kgrid_g(1.0);
    // x[g][i] = E[k^gamma | counts]
    let x: Vec<Vec<f64>> = gs.iter().map(|&g| {
        let kp: Vec<f64> = kg.iter().map(|&k| if k > 0.0 { k.powf(g) } else { 0.0 }).collect();
        (0..n).map(|i| (0..NK).map(|j| wpost[i * NK + j] * kp[j]).sum()).collect()
    }).collect();
    let iv: Vec<f64> = vc.iter().map(|&v| 1.0 / v).collect();
    // sufficient sums per (gamma, fold) for a given cell multiplicity c_i
    let fit = |c: &[f64]| -> usize {
        let mut best = (f64::INFINITY, 0);
        for gi in 0..ng {
            let mut rss = 0.0;
            for f in 0..2 {
                let (mut sxx, mut sxu, mut suu) = (0.0, 0.0, 0.0);
                for i in 0..n {
                    if fold[i] != f || c[i] == 0.0 {
                        continue;
                    }
                    let w = c[i] * iv[i];
                    sxx += w * x[gi][i] * x[gi][i];
                    sxu += w * x[gi][i] * u[i];
                    suu += w * u[i] * u[i];
                }
                let m = if sxx > 0.0 { (sxu / sxx).max(0.0) } else { 0.0 };
                rss += suu - 2.0 * m * sxu + m * m * sxx;
            }
            if rss < best.0 {
                best = (rss, gi);
            }
        }
        best.1
    };
    let ones = vec![1f64; n];
    let g0 = fit(&ones);
    let mut st = name.bytes().fold(0x5CA1Eu64, |h, b| h.wrapping_mul(131).wrapping_add(b as u64));
    let byf: [Vec<usize>; 2] = [(0..n).filter(|&i| fold[i] == 0).collect(), (0..n).filter(|&i| fold[i] == 1).collect()];
    let nb = 200;
    let mut lg: Vec<f64> = Vec::with_capacity(nb);
    let mut c = vec![0f64; n];
    for _ in 0..nb {
        c.iter_mut().for_each(|v| *v = 0.0);
        for f in 0..2 {
            let m = byf[f].len() as u64;
            for _ in 0..m {
                c[byf[f][(splitmix(&mut st) % m) as usize]] += 1.0;
            }
        }
        lg.push(gs[fit(&c)].ln());
    }
    lg.sort_by(f64::total_cmp);
    let lo = lg[(0.025 * nb as f64) as usize].exp();
    let hi = lg[((0.975 * nb as f64) as usize).min(nb - 1)].exp();
    let above = lg.iter().filter(|&&v| v > 0.0).count() as f64 / nb as f64;
    let below = lg.iter().filter(|&&v| v < 0.0).count() as f64 / nb as f64;
    [gs[g0], lo, hi, (2.0 * above.min(below)).min(1.0)]
}

/// Solve a small symmetric positive-definite system (Cholesky).
fn solve_spd<const D: usize>(a: &[[f64; D]; D], b: &[f64; D]) -> Option<[f64; D]> {
    let mut l = [[0f64; D]; D];
    for i in 0..D {
        for j in 0..=i {
            let mut s = a[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                l[i][i] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    let mut y = [0f64; D];
    for i in 0..D {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i][k] * y[k];
        }
        y[i] = s / l[i][i];
    }
    let mut x = [0f64; D];
    for i in (0..D).rev() {
        let mut s = y[i];
        for k in i + 1..D {
            s -= l[k][i] * x[k];
        }
        x[i] = s / l[i][i];
    }
    Some(x)
}

fn nb_table(cnt: &[f64], lib: &[f64], a: f64, r: f64, gamma: f64) -> Vec<f64> {
    let kg = kgrid_g(gamma);
    let n = cnt.len();
    let mut t = vec![0f64; n * NK];
    for i in 0..n {
        for j in 0..NK {
            let mu = (lib[i] * a * (1.0 - kg[j])).max(1e-12);
            t[i * NK + j] = if r.is_finite() {
                r * (r / (r + mu)).ln() + cnt[i] * (mu / (r + mu)).ln()
            } else {
                -mu + cnt[i] * mu.ln()
            };
        }
    }
    t
}

fn target_counts(c: &Cache, cells: &[u32], tg: usize) -> (Vec<f64>, Vec<f64>) {
    let mut cnt = Vec::with_capacity(cells.len());
    let mut lib = Vec::with_capacity(cells.len());
    for &r in cells {
        let (ix, ct) = c.row(r as usize);
        let k = match ix.binary_search(&(tg as u16)) {
            Ok(p) => ct[p] as f64,
            Err(_) => 0.0,
        };
        let s = c.scale[r as usize];
        cnt.push(k);
        lib.push(if s > 0.0 { 1e4 / s as f64 } else { 0.0 });
    }
    (cnt, lib)
}

fn one_joint(c: &Cache, rf: &Ref, gid: &HashMap<&str, usize>, name: &str, all: &[u32], prm: &JParams) -> JRes {
    let cells: Vec<u32> = match prm.half {
        None => all.to_vec(),
        Some(h) => all.iter().cloned().filter(|&r| (hash(r) & 1) as u32 == h).collect(),
    };
    let n = cells.len();
    if n < prm.min_cells {
        return JRes::empty(name, n, "few_cells");
    }
    let fold: Vec<usize> = cells.iter().map(|&r| ((hash(r) >> 1) & 1) as usize).collect();
    let na = fold.iter().filter(|&&f| f == 0).count();
    if na < 10 || n - na < 10 {
        return JRes::empty(name, n, "small_fold");
    }
    let tg = gid.get(name.rsplit('|').next().unwrap()).copied();
    let drop_g = tg.and_then(|g| if rf.gpos[g] >= 0 { Some(rf.gpos[g] as usize) } else { None });
    let tp = Instant::now();
    let mut d = dense(c, rf, &cells, drop_g);
    prof_add(6, tp);
    let sel: [Vec<usize>; 2] = [(0..n).filter(|&i| fold[i] == 0).collect(), (0..n).filter(|&i| fold[i] == 1).collect()];
    // target NB parameters from reference controls
    let (mut cnt, lib, a, r) = match tg {
        Some(g) => {
            let a = rf.s_c[g] / rf.s_l;
            let ss = rf.s_cc[g] - 2.0 * a * rf.s_cl[g] + a * a * rf.s_ll; // sum (c - L a)^2
            let inv_r = (ss - a * rf.s_l) / (a * a * rf.s_ll);
            let r = if inv_r > 1e-6 { 1.0 / inv_r } else { f64::INFINITY };
            let (cn, lb) = target_counts(c, &cells, g);
            (cn, lb, a, r)
        }
        None => (vec![0.0; n], vec![0.0; n], 0.0, f64::INFINITY),
    };
    let tmeas = a > 0.0;
    let mut sim_k: Option<Vec<f64>> = None;
    let mut dump_k: Option<Vec<f64>> = None;
    let tmean = a * rf.s_l / rf.n_ref as f64;
    if let (Some(sp), true) = (&prm.sim, name.starts_with("NULL_")) {
        let ks = simulate(sp, name, &mut d, &mut cnt, drop_g);
        dump_k = Some(ks.clone());
        if sp.pure == 'p' || sp.pure == 'c' {
            let mut st = 0xBADA55u64 ^ hash(n as u32);
            for i in 0..n {
                let mu = lib[i] * a * (1.0 - ks[i]);
                let lam = if r.is_finite() { rgamma(&mut st, r) * mu / r } else { mu };
                cnt[i] = rpois(&mut st, lam);
            }
        }
        if sp.pure == 'p' || sp.pure == 's' {
            sim_k = Some(ks);
        }
    }
    let use_t = tmeas && !prm.no_target;
    let _ = &sim_k;
    let zero_t = if use_t { vec![] } else { vec![0.0; n * NK] };

    let mut res = JRes::empty(name, n, "ok");
    res.na = na;
    res.nbb = n - na;
    res.tmeas = tmeas;
    res.tmean = tmean;

    let bt: Vec<usize> = cells.iter().map(|&r| batch_of(c, r)).collect();
    let lls: Vec<f64> = cells.iter().map(|&r| loglib(c, r)).collect();
    let mut h = vec![1f64; n];
    let gg = ggrid();
    let mid = gg.len() / 2; // gamma = 1
    let mut u = vec![0f64; n];
    let mut v = [1f64; 2];
    let mut vc = vec![1f64; n];
    let mut best = (f64::NEG_INFINITY, 0usize);
    let mut prof = vec![0f64; gg.len()];
    let mut post_k = vec![0f64; n];
    let mut post_h = vec![0f64; n];
    let mut fit_pi = [0f64; NK];
    let mut fit_m = [0f64; 2];
    let mut warm: Option<EmState> = None;
    let mut fit_state: Option<EmState> = None;
    let mut cs_fold: [Option<CStat>; 2] = [None, None];
    let mut lin_state: Option<EmState> = None;
    let mut thk: Option<[f64; DP]> = None;
    for it in 0..=prm.outer {
        // cross-fitted scores: beta from fold f scores fold 1 - f
        let mut tf = [0f64; 2];
        let tp = Instant::now();
        for f in 0..2 {
            let (w, _, _) = fit_beta(&d, rf, &sel[f], &h);
            let wc = wconst(rf, &w);
            let cs = ctrl_score(rf, &w);
            // per cell variance: cell noise + error of the per batch centring
            // (shared within a batch, var sw2 / m_b) + error of the depth slope
            let mut sv = 0.0;
            for &i in &sel[1 - f] {
                vc[i] = cs.var(bt[i], lls[i]);
                sv += vc[i];
            }
            v[1 - f] = sv / sel[1 - f].len() as f64;
            // variance of the fold sum: cell noise + noise of the per batch centring
            let mut nfb = vec![0f64; rf.nb];
            let mut sf = 0.0;
            for &i in &sel[1 - f] {
                u[i] = cs.adj(score(&d, i, &w, &wc), bt[i], lls[i]);
                nfb[bt[i]] += 1.0;
                sf += u[i];
            }
            let var = sel[1 - f].len() as f64 * cs.sw2 + (0..rf.nb).map(|b| nfb[b] * nfb[b] * cs.sw2 / cs.mb[b]).sum::<f64>();
            tf[1 - f] = sf / var.sqrt();
            cs_fold[1 - f] = Some(cs);
        }
        prof_add(0, tp);
        let tp = Instant::now();
        if let (Ok(pfx), true) = (std::env::var("DOSE_DUMP"), it == 0) {
            let ks = dump_k.clone().unwrap_or_else(|| vec![-1.0; n]);
            let mut f = BufWriter::new(File::create(format!("{pfx}_{}.tsv", name.replace('|', "_"))).unwrap());
            writeln!(f, "fold\tu\tv\tk\tL\tcnt\tbatch").unwrap();
            for i in 0..n {
                writeln!(f, "{}\t{}\t{}\t{}\t{}\t{}\t{}", fold[i], u[i], vc[i], ks[i], lib[i], cnt[i], bt[i]).unwrap();
            }
        }
        if let (Some(ks), Some(sp)) = (&sim_k, &prm.sim) {
            // pure model: u_i = m h_i + N(0, v), per cell SNR 3
            let mut st = 0x5EEDu64 ^ hash(n as u32);
            for i in 0..n {
                let hi = if ks[i] > 0.0 { ks[i].powf(sp.gamma) } else { 0.0 };
                u[i] = 3.0 * v[fold[i]].sqrt() * hi + vc[i].sqrt() * std_normal(&mut st);
            }
        }
        if it == 0 {
            // calibrated phenotype test (no selection: beta never saw the scored cells)
            // T_A and T_B reuse the same cells in opposite roles and are positively
            // correlated; their average has variance <= 1 (conservative).
            let z = 0.5 * (tf[0] + tf[1]);
            res.t = z;
            res.p = norm_sf(z);
        }
        // profile over gamma, warm started outwards from gamma = 1. Earlier
        // outer passes only need the response weights: coarse grid.
        // Without target counts the likelihood does not depend on gamma.
        let idx: Vec<usize> = if !use_t {
            vec![mid]
        } else if it < prm.outer {
            vec![mid, mid + 4, mid + 8, mid - 4, mid - 8]
        } else {
            let mut v: Vec<usize> = (mid..gg.len()).collect();
            v.extend((0..mid).rev());
            v
        };
        if anchor_on() && use_t && thk.is_none() {
            let nbl = nb_table(&cnt, &lib, a, r, 1.0);
            let (_, _, _, sk) = with_anchor(None, || em(&vec![0f64; n], &fold, &vec![1f64; n], &nbl, Some(EmState { p0: 0.1, th: [0.0; DP], m: [0.0; 2], t: 0.0 }), em_iters()));
            thk = Some(sk.th);
        }
        best = (f64::NEG_INFINITY, mid);
        prof.iter_mut().for_each(|x| *x = f64::NAN);
        let mut best_w = vec![];
        let mut best_pi = [0f64; NK];
        let mut st_at: Vec<Option<EmState>> = vec![None; gg.len()];
        for &gi in &idx {
            // warm start from the neighbouring gamma value already fitted
            let init = if gi == mid { warm } else if gi > mid { st_at[gi - 1].or(st_at[mid]) } else { st_at[gi + 1].or(st_at[mid]) };
            let init = init.or_else(|| { let mut k = gi; while k != mid { k = if k > mid { k - 1 } else { k + 1 }; if st_at[k].is_some() { return st_at[k]; } } None });
            let tab;
            let nbl: &[f64] = if use_t { tab = nb_table(&cnt, &lib, a, r, gg[gi]); &tab } else { &zero_t };
            let init = init.map(|mut s| { if let Some(t) = thk { s.th = t; } s });
            let (ll, pi, w, st) = with_anchor(thk.map(|t| (gg[gi], t)), || em(&u, &fold, &vc, nbl, init, em_iters()));
            prof[gi] = ll;
            st_at[gi] = Some(st);
            if ll > best.0 {
                best = (ll, gi);
                best_w = w;
                best_pi = pi;
            }
        }
        prof_add(1, tp);
        if !use_t {
            prof.iter_mut().for_each(|x| *x = best.0);
        }
        warm = st_at[mid];
        fit_state = st_at[best.1];
        lin_state = st_at[mid];
        let (pi, m, w) = (&best_pi, &st_at[best.1].unwrap().m, &best_w);
        let kg = kgrid_g(gg[best.1]);
        let hg = hgrid();
        for i in 0..n {
            let (mut ek, mut eh) = (0.0, 0.0);
            for j in 0..NK {
                ek += w[i * NK + j] * kg[j];
                eh += w[i * NK + j] * hg[j];
            }
            post_k[i] = ek;
            post_h[i] = eh;
        }
        fit_pi = *pi;
        fit_m = *m;
        if it < prm.outer {
            // next pass: weight cells by their posterior response
            h.copy_from_slice(&post_h);
            if h.iter().map(|x| x * x).sum::<f64>() < 1e-6 {
                break;
            }
        }
    }
    let ll = best.0;
    // robust (sandwich) scaling of the profile likelihood ratio
    let mut ratio = 1.0;
    let prof_raw = prof.clone();
    if use_t && robust_on() {
        let st = fit_state.unwrap();
        let x = LikIn { u: &u, fold: &fold, vc: &vc, cnt: &cnt, lib: &lib, a, r };
        let p0 = st.p0.clamp(1e-6, 1.0 - 1e-6);
        let th = [gg[best.1].ln(), st.m[0], st.m[1], (p0 / (1.0 - p0)).ln(), st.th[0], st.th[1], st.th[2], st.th[3]];
        ratio = robust_ratio(&x, &th).min(1.0);
        for x in prof.iter_mut() {
            *x = ll - ratio * (ll - *x);
        }
    }
    res.rratio = ratio;
    res.tau2 = fit_state.map_or(f64::NAN, |s| s.t);
    if use_t {
        // identifiability: dose distribution from the target counts alone
        // (scores switched off, so response heterogeneity cannot widen it);
        // dose_mid = prior mass with knockdown in [0.2, 0.8]
        let tp = Instant::now();
        let u0 = vec![0f64; n];
        let v0 = vec![1f64; n];
        let nbl = nb_table(&cnt, &lib, a, r, 1.0);
        let (_, pi, wpost, _) = em(&u0, &fold, &v0, &nbl, Some(EmState { p0: 0.1, th: [0.0; DP], m: [0.0; 2], t: 0.0 }), em_iters());
        let hg = hgrid();
        res.dose_mid = (0..NK).filter(|&j| hg[j] >= 0.2 && hg[j] <= 0.8).map(|j| pi[j]).sum();
        prof_add(2, tp);
        let tp = Instant::now();
        res.rc = regcal(&u, &fold, &vc, &wpost, name);
        prof_add(3, tp);
    }
    // likelihood ratio statistic for gamma = 1 (linear dose response)
    res.lr1 = 2.0 * (ll - prof[mid]);
    if use_t && n_boot() > 0 {
        // observed LR on the same 9 point grid the bootstrap uses
        let best9 = (0..gg.len()).filter(|&i| (i as i64 - mid as i64) % 2 == 0).map(|i| prof_raw[i]).fold(f64::NEG_INFINITY, f64::max);
        let raw = 2.0 * (best9 - prof_raw[mid]);
        if raw > 3.84 {
            let (tc, _) = target_counts(c, &rf.cv_rows, tg.unwrap());
            let cs = [cs_fold[0].take().unwrap(), cs_fold[1].take().unwrap()];
            res.boot = Some(Box::new(BootJob { fold: fold.clone(), cs, tc, lin: lin_state.unwrap(), a, r, raw }));
        } else {
            res.p_lin = 1.0;
        }
    }
    let lo = (0..gg.len()).find(|&i| prof[i] >= ll - 1.92).unwrap_or(0);
    let hi = (0..gg.len()).rev().find(|&i| prof[i] >= ll - 1.92).unwrap_or(gg.len() - 1);
    res.gamma = gg[best.1];
    // CI end at the grid edge means unbounded on that side
    res.glo = if lo == 0 { 0.0 } else { gg[lo] };
    res.ghi = if hi == gg.len() - 1 { f64::INFINITY } else { gg[hi] };
    res.ll_range = ll - prof.iter().cloned().fold(f64::INFINITY, f64::min);
    res.m = fit_m;
    res.v = v;
    res.pi0 = fit_pi[0];
    res.mean_k = post_k.iter().sum::<f64>() / n as f64;
    // per-gene DE with all cells and posterior response
    let alls: Vec<usize> = (0..n).collect();
    // per-gene DE: plain mean shift (h = 1) with adaptive shrinkage. Using the
    // posterior dose here would reuse the cells that produced it.
    let ones = vec![1f64; n];
    let tp = Instant::now();
    let (_, lfsr, _) = fit_beta(&d, rf, &alls, &ones);
    prof_add(5, tp);
    res.n_de = lfsr.iter().filter(|&&x| x < 0.05).count();
    if prm.write_cells {
        res.cells = (0..n).map(|i| (cells[i], post_k[i], post_h[i])).collect();
    }
    res
}

fn one_mixscale(c: &Cache, rf: &Ref, ctrl_ref: &[u32], gid: &HashMap<&str, usize>, name: &str, all: &[u32], prm: &JParams) -> JRes {
    let cells: Vec<u32> = match prm.half {
        None => all.to_vec(),
        Some(h) => all.iter().cloned().filter(|&r| (hash(r) & 1) as u32 == h).collect(),
    };
    let n = cells.len();
    if n < prm.min_cells {
        return JRes::empty(name, n, "few_cells");
    }
    let tg = gid.get(name.rsplit('|').next().unwrap()).copied();
    let drop_g = tg.and_then(|g| if rf.gpos[g] >= 0 { Some(rf.gpos[g] as usize) } else { None });
    let d = dense(c, rf, &cells, drop_g);
    let g = d.g;
    // mean shift on the same cells, t statistic vs 0 (controls are centred)
    let alls: Vec<usize> = (0..n).collect();
    let (sx, sxx, _) = colsum(&d, rf, &alls, &vec![1f64; n]);
    let nf = n as f64;
    let delta: Vec<f64> = sx.iter().map(|s| s / nf).collect();
    let tt: Vec<f64> = (0..g)
        .map(|j| {
            let var = ((sxx[j] - nf * delta[j] * delta[j]) / (nf - 1.0)).max(1e-6);
            delta[j] / (var / nf + 1.0 / rf.n_ref as f64).sqrt()
        })
        .collect();
    // Bonferroni DE; fall back to the top 20 genes like Mixscape/Mixscale
    let zc = {
        // two-sided 0.05 / G
        let p = 0.05 / g as f64;
        let (mut lo, mut hi): (f64, f64) = (0.0, 40.0);
        for _ in 0..100 {
            let mid = 0.5 * (lo + hi);
            if 2.0 * norm_sf(mid) > p { lo = mid } else { hi = mid }
        }
        hi
    };
    let mut de: Vec<usize> = (0..g).filter(|&j| tt[j].abs() > zc && Some(j) != drop_g).collect();
    let n_de = de.len();
    if de.len() < 5 {
        let mut ord: Vec<usize> = (0..g).filter(|&j| Some(j) != drop_g).collect();
        ord.sort_by(|&a, &b| tt[b].abs().total_cmp(&tt[a].abs()));
        de = ord[..20.min(ord.len())].to_vec();
    }
    let norm: f64 = de.iter().map(|&j| delta[j] * delta[j]).sum::<f64>().sqrt().max(1e-12);
    let mut w = vec![0f64; g];
    for &j in &de {
        w[j] = delta[j] / norm;
    }
    let cs = ctrl_score(rf, &w);
    let wc = wconst(rf, &w);
    let csd = cs.sw2_raw.sqrt();
    let z: Vec<f64> = (0..n).map(|i| (score(&d, i, &w, &wc) - cs.cmb[batch_of(c, cells[i])]) / csd).collect();
    let mut res = JRes::empty(name, n, "ok");
    res.n_de = n_de;
    res.t = z.iter().sum::<f64>() / nf.sqrt();
    res.p = norm_sf(res.t);
    // dose axis: target expression, linear, 0 = min, 1 = control median
    let Some(t) = tg else {
        res.status = "target_not_in_genes";
        return res;
    };
    let (cnt, lib) = target_counts(c, &cells, t);
    let e: Vec<f64> = cnt.iter().zip(&lib).map(|(&k, &l)| if l > 0.0 { 1e4 * k / l } else { 0.0 }).collect();
    let (ccnt, clib) = target_counts(c, ctrl_ref, t);
    let mut ce: Vec<f64> = ccnt.iter().zip(&clib).map(|(&k, &l)| if l > 0.0 { 1e4 * k / l } else { 0.0 }).collect();
    ce.sort_by(f64::total_cmp);
    let med = {
        let h = (ce.len() - 1) as f64 * 0.5;
        let lo = h.floor() as usize;
        let hi = (lo + 1).min(ce.len() - 1);
        ce[lo] + (h - lo as f64) * (ce[hi] - ce[lo])
    };
    let mn = e.iter().cloned().fold(ce[0], f64::min);
    res.tmeas = med > mn;
    if med <= mn {
        res.status = "ctrl_median_le_min";
        return res;
    }
    let xs: Vec<f64> = e.iter().map(|&v| (v - mn) / (med - mn)).collect();
    let zmax = z.iter().cloned().fold(0.0, f64::max).max(1e-9);
    let ps: Vec<f64> = z.iter().map(|&v| v.max(0.0) / zmax).collect();
    let bins = ps::make_bins(&xs, &ps, 10, ps::BinMode::Rank);
    res.nbins = bins.len();
    let bx: Vec<f64> = bins.iter().map(|b| b.0).collect();
    let by: Vec<f64> = bins.iter().map(|b| b.1).collect();
    match hill::fit_ll4(&bx, &by, None) {
        Some(f) => {
            res.ed50 = f.e;
            res.h0 = hill::predict(&f, 0.0);
            res.h1 = hill::predict(&f, 1.0);
            res.keep = res.h0 - res.h1 >= 0.2;
        }
        None => res.status = "fit_failed",
    }
    if prm.write_cells {
        res.cells = (0..n).map(|i| (cells[i], xs[i], ps[i])).collect();
    }
    res
}

pub fn run(dir: &str, out: &str, prm: &JParams, only: Option<Vec<String>>) -> Result<()> {
    let t0 = Instant::now();
    let c = Cache::open(dir)?;
    // same control split and mock groups as `dosecurve ps --null`
    let ctrl_all = c.cells(0);
    let (ref_c, hold): (Vec<u32>, Vec<u32>) = if prm.null > 0 {
        ctrl_all.iter().partition(|&&r| {
            let mut z = r as u64 ^ 0x5DEECE66D;
            splitmix(&mut z) % 4 != 0
        })
    } else {
        (ctrl_all.to_vec(), vec![])
    };
    let rf = build_ref(&c, &ref_c, prm);
    eprintln!("[joint] reference controls {} (held out {}), modelled genes {}, batches {}, {:.1}s", ref_c.len(), hold.len(), rf.gsel.len(), rf.nb, t0.elapsed().as_secs_f64());
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
    // largest groups first for better load balance
    let mut order: Vec<usize> = (0..jobs.len()).collect();
    order.sort_by(|&a, &b| jobs[b].1.len().cmp(&jobs[a].1.len()));
    let mut res: Vec<(usize, JRes)> = order
        .par_iter()
        .map(|&k| {
            let (n, cl) = &jobs[k];
            let r = match prm.method {
                Method::Joint => one_joint(&c, &rf, &gid, n, cl, prm),
                Method::Mixscale => one_mixscale(&c, &rf, &ref_c, &gid, n, cl, prm),
            };
            (k, r)
        })
        .collect();
    res.sort_by_key(|x| x.0);
    let res: Vec<JRes> = res.into_iter().map(|x| x.1).collect();
    let mut res = res;
    // phase 2: linear-null bootstrap for the flagged groups, largest first
    let tb = Instant::now();
    let mut bj: Vec<usize> = (0..res.len()).filter(|&i| res[i].boot.is_some()).collect();
    bj.sort_by(|&x, &y| res[y].n.cmp(&res[x].n));
    let pl: Vec<(usize, f64)> = bj.par_iter().map(|&i| {
        let b = res[i].boot.as_ref().unwrap();
        (i, boot_linear(&b.fold, [&b.cs[0], &b.cs[1]], &b.tc, b.lin, b.a, b.r, b.raw, n_boot(), &res[i].name))
    }).collect();
    for (i, p) in pl {
        res[i].p_lin = p;
        res[i].boot = None;
    }
    PROF[4].fetch_add(tb.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    eprintln!("[joint] {} groups in {:.1}s (bootstrap {} groups, {:.1}s wall)", res.len(), t1.elapsed().as_secs_f64(), bj.len(), tb.elapsed().as_secs_f64());
    let lab = ["scores", "gamma profile", "dose_mid EM", "regcal", "bootstrap", "n_de fit_beta", "dense"];
    for (k, l) in lab.iter().enumerate() {
        eprintln!("[prof] {l}: {:.1} thread-s", PROF[k].load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9);
    }
    let (a, b) = (EMCALL.load(std::sync::atomic::Ordering::Relaxed), EMIT.load(std::sync::atomic::Ordering::Relaxed));
    eprintln!("[prof] EM calls {a}, iterations {b}, mean {:.1}", b as f64 / a.max(1) as f64);

    // BH over real perturbations; mock groups use the same p threshold
    let mut ps_real: Vec<f64> = res.iter().filter(|r| !r.name.starts_with("NULL_") && r.p.is_finite()).map(|r| r.p).collect();
    ps_real.sort_by(f64::total_cmp);
    let m = ps_real.len();
    let mut thr = 0.0;
    for (i, &p) in ps_real.iter().enumerate() {
        if p <= 0.05 * (i + 1) as f64 / m as f64 {
            thr = p;
        }
    }
    let mut w = BufWriter::new(File::create(format!("{out}.tsv"))?);
    match prm.method {
        Method::Joint => writeln!(w, "gene\tn_cells\tn_A\tn_B\ttarget_measured\ttarget_ctrl_mean\tT\tp\tpheno\tn_de_lfsr\tm_A\tm_B\tv_A\tv_B\tpi0\tmean_k\tgamma\tgamma_lo\tgamma_hi\tr50\tll_range\trobust_ratio\tlr_gamma1\tp_linear\ttau2\tdose_mid\tgamma_rc\tgamma_rc_lo\tgamma_rc_hi\tclass_rc\tclass\tstatus")?,
        Method::Mixscale => writeln!(w, "gene\tn_cells\tn_de\tT\tp\tpheno\ttarget_measured\tn_bins\ted50\thill0\thill1\tdelta\tkeep\tclass\tstatus")?,
    }
    let (mut nc, mut nbuf, mut nsen, mut nind) = (0, 0, 0, 0);
    for r in &res {
        // simulation runs have no real perturbations: nominal 0.05
        let pheno = r.p.is_finite() && if prm.sim.is_some() { r.p <= 0.05 } else { r.p <= thr && m > 0 };
        match prm.method {
            Method::Joint => {
                let class = if r.status != "ok" || !pheno {
                    "NA"
                } else if r.p_lin.is_finite() {
                    if r.p_lin < 0.05 && r.gamma > 1.0 { "buffered" } else if r.p_lin < 0.05 && r.gamma < 1.0 { "sensitive" } else { "indeterminate" }
                } else if r.glo > 1.0 {
                    "buffered"
                } else if r.ghi < 1.0 {
                    "sensitive"
                } else {
                    "indeterminate"
                };
                match class {
                    "buffered" => nbuf += 1,
                    "sensitive" => nsen += 1,
                    "indeterminate" => nind += 1,
                    _ => {}
                }
                if class != "NA" {
                    nc += 1;
                }
                let r50 = 1.0 - 0.5f64.powf(1.0 / r.gamma);
                let class_rc = if r.status != "ok" || !pheno || !r.rc[0].is_finite() { "NA" } else if r.rc[1] > 1.0 { "buffered" } else if r.rc[2] < 1.0 { "sensitive" } else { "indeterminate" };
                writeln!(w, "{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4e}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.3}\t{:.3}\t{:.3}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}\t{}\t{}",
                    r.name, r.n, r.na, r.nbb, r.tmeas as u8, r.tmean, r.t, r.p, pheno as u8, r.n_de, r.m[0], r.m[1], r.v[0], r.v[1], r.pi0, r.mean_k, r.gamma, r.glo, r.ghi, r50, r.ll_range, r.rratio, r.lr1, r.p_lin, r.tau2, r.dose_mid, r.rc[0], r.rc[1], r.rc[2], class_rc, class, r.status)?;
            }
            Method::Mixscale => {
                let class = if r.keep { if r.ed50 <= 0.5 { "buffered" } else { "sensitive" } } else { "NA" };
                match class {
                    "buffered" => nbuf += 1,
                    "sensitive" => nsen += 1,
                    _ => {}
                }
                if class != "NA" {
                    nc += 1;
                }
                writeln!(w, "{}\t{}\t{}\t{:.4}\t{:.4e}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}\t{}\t{}",
                    r.name, r.n, r.n_de, r.t, r.p, pheno as u8, r.tmeas as u8, r.nbins, r.ed50, r.h0, r.h1, r.h0 - r.h1, r.keep as u8, class, r.status)?;
            }
        }
    }
    if prm.write_cells {
        let mut wc = BufWriter::new(File::create(format!("{out}.cells.tsv"))?);
        writeln!(wc, "gene\tcell_row\tk_or_x\th_or_score")?;
        for r in &res {
            for &(cell, a, b) in &r.cells {
                writeln!(wc, "{}\t{}\t{:.5}\t{:.5}", r.name, cell, a, b)?;
            }
        }
    }
    eprintln!("[joint] BH p threshold {:.3e}; classified {nc} (buffered {nbuf}, sensitive {nsen}, indeterminate {nind}); total {:.1}s", thr, t0.elapsed().as_secs_f64());
    Ok(())
}
