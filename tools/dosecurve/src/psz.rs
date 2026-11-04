//! Faithful Song et al. 2025 mode on Replogle normalised (z) expression.
//! Mirrors scmageck_eff_estimate run on a Seurat object whose `data` layer is
//! the normalised matrix and `counts` layer the UMI counts:
//!  - FindMarkers (Seurat 5, wilcox.test fallback): min.pct 0.01 on data > 0,
//!    log2FC = log2(mean(expm1(x)) + 1) difference, |log2FC| >= 0.1
//!    (x0.8 up to 3 rounds if < 10 genes), order by p then -log2FC, top 500
//!  - Y genes: counts > 0 in >= 1% of cells; clip at 95th percentile (type 7)
//!  - ridge 0.01 on [1, 1{P}], per-cell x in [0, 3] (closed form of the
//!    L-BFGS-B problem), /3, / max over all cells
//!  - dose: z of the perturbed gene, 0 = min over cells, 1 = control median;
//!    cut(x, unique(quantile(x, 0:10/10)), include.lowest) bins; LL.4.
//! The normalised X is mmapped straight from the h5ad (contiguous f32).

use crate::hill;
use crate::ps::{BinMode, Cache, make_bins};
use anyhow::Result;
use memmap2::Mmap;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

pub struct Norm {
    map: Mmap,
    off: usize,
    pub ng: usize,
    pub genes: Vec<String>,
    pub rawcol: Vec<u32>,
    pub groups: Vec<(String, usize, usize)>,
    members: Vec<u32>,
}

impl Norm {
    pub fn open(h5ad: &str, meta: &str) -> Result<Norm> {
        let d = Path::new(meta);
        let m: Vec<usize> = std::fs::read_to_string(d.join("meta.txt"))?.split_whitespace().map(|v| v.parse().unwrap()).collect();
        let f = File::open(h5ad)?;
        let map = unsafe { Mmap::map(&f)? };
        let groups = std::fs::read_to_string(d.join("groups.tsv"))?
            .lines()
            .map(|l| {
                let f: Vec<&str> = l.split('\t').collect();
                (f[0].to_string(), f[1].parse().unwrap(), f[2].parse().unwrap())
            })
            .collect();
        Ok(Norm {
            map,
            off: m[0],
            ng: m[2],
            genes: std::fs::read_to_string(d.join("genes.txt"))?.lines().map(String::from).collect(),
            rawcol: bytemuck::cast_slice::<u8, u32>(&std::fs::read(d.join("rawcol.bin"))?).to_vec(),
            groups,
            members: bytemuck::cast_slice::<u8, u32>(&std::fs::read(d.join("members.bin"))?).to_vec(),
        })
    }
    #[inline]
    pub fn row(&self, r: usize) -> &[f32] {
        let b = &self.map[self.off + r * self.ng * 4..self.off + (r + 1) * self.ng * 4];
        bytemuck::cast_slice(b)
    }
    pub fn cells(&self, g: usize) -> &[u32] {
        let (_, o, n) = self.groups[g];
        &self.members[o..o + n]
    }
}

struct Ctrl {
    n: usize,
    gm: Vec<f32>,       // gene-major values, ng x n
    sorted: Vec<f32>,   // per gene sorted
    tie: Vec<f64>,      // per gene sum(t^3 - t) over control tie groups
    sum: Vec<f64>,      // per gene sum of values
    lin: Vec<f64>,      // per gene mean expm1
    pos: Vec<u32>,      // per gene count of data > 0
    rawnz: Vec<u32>,    // per normalised gene count of counts > 0
    median: Vec<f32>,
    minv: Vec<f32>,
}

fn build_ctrl(nm: &Norm, raw: &Cache, cells: &[u32]) -> Ctrl {
    let (n, ng) = (cells.len(), nm.ng);
    let mut gm = vec![0f32; ng * n];
    for (i, &r) in cells.iter().enumerate() {
        let row = nm.row(r as usize);
        for g in 0..ng {
            gm[g * n + i] = row[g];
        }
    }
    let mut sorted = gm.clone();
    sorted.par_chunks_mut(n).for_each(|s| s.sort_unstable_by(f32::total_cmp));
    let stats: Vec<(f64, f64, f64, u32, f32)> = (0..ng)
        .into_par_iter()
        .map(|g| {
            let s = &sorted[g * n..(g + 1) * n];
            let mut tie = 0f64;
            let mut i = 0;
            while i < n {
                let mut j = i + 1;
                while j < n && s[j] == s[i] {
                    j += 1;
                }
                let t = (j - i) as f64;
                tie += t * t * t - t;
                i = j;
            }
            let sum: f64 = s.iter().map(|&v| v as f64).sum();
            let lin: f64 = s.iter().map(|&v| (v as f64).exp_m1()).sum::<f64>() / n as f64;
            let pos = s.iter().filter(|&&v| v > 0.0).count() as u32;
            let h = (n - 1) as f64 * 0.5;
            let lo = h.floor() as usize;
            let med = s[lo] + (h - lo as f64) as f32 * (s[(lo + 1).min(n - 1)] - s[lo]);
            (tie, sum, lin, pos, med)
        })
        .collect();
    // counts > 0 per raw gene in control cells
    let nraw = raw.genes.len();
    let mut rnz = vec![0u32; nraw];
    for &r in cells {
        for &g in raw.row(r as usize).0 {
            rnz[g as usize] += 1;
        }
    }
    let rawnz = nm.rawcol.iter().map(|&j| rnz[j as usize]).collect();
    let minv = (0..ng).map(|g| sorted[g * n]).collect();
    Ctrl {
        n,
        gm,
        sorted,
        tie: stats.iter().map(|s| s.0).collect(),
        sum: stats.iter().map(|s| s.1).collect(),
        lin: stats.iter().map(|s| s.2).collect(),
        pos: stats.iter().map(|s| s.3).collect(),
        median: stats.iter().map(|s| s.4).collect(),
        rawnz,
        minv,
    }
}

pub struct Res {
    pub name: String,
    pub n_cells: usize,
    pub n_de: usize,
    pub n_target: usize,
    pub targets: Vec<usize>,
    pub ed: Option<hill::Fit>,
    pub bins: Vec<(f64, f64, usize)>,
    pub cells: Vec<(u32, f32, f32)>,
    pub status: &'static str,
}

/// 2 * Phi(-|z|) ordering key: R's p-value underflows to 0 beyond |z| ~ 38.5
fn pkey(z: f64) -> f64 {
    let a = z.abs();
    if a > 38.4 { f64::INFINITY } else { a }
}

fn one(nm: &Norm, raw: &Cache, c: &Ctrl, name: &str, cells: &[u32], min_cells: usize, mode: BinMode) -> Res {
    let (ng, nc, np) = (nm.ng, c.n, cells.len());
    let mut res = Res { name: name.into(), n_cells: np, n_de: 0, n_target: 0, targets: vec![], ed: None, bins: vec![], cells: vec![], status: "ok" };
    if np < min_cells {
        res.status = "few_cells";
        return res;
    }
    // pert block gene-major
    let mut pm = vec![0f32; ng * np];
    for (i, &r) in cells.iter().enumerate() {
        let row = nm.row(r as usize);
        for g in 0..ng {
            pm[g * np + i] = row[g];
        }
    }
    let nraw = raw.genes.len();
    let mut rnz = vec![0u32; nraw];
    for &r in cells {
        for &g in raw.row(r as usize).0 {
            rnz[g as usize] += 1;
        }
    }
    // DE
    let mut lfc = vec![f64::NAN; ng];
    for g in 0..ng {
        let p = &pm[g * np..(g + 1) * np];
        let pct1 = p.iter().filter(|&&v| v > 0.0).count() as f64 / np as f64;
        let pct2 = c.pos[g] as f64 / nc as f64;
        if pct1.max(pct2) < 0.01 {
            continue;
        }
        let m1 = p.iter().map(|&v| (v as f64).exp_m1()).sum::<f64>() / np as f64;
        lfc[g] = (m1 + 1.0).log2() - (c.lin[g] + 1.0).log2();
    }
    let mut thr = 0.1;
    let mut de: Vec<usize>;
    let mut round = 0;
    loop {
        de = (0..ng).filter(|&g| lfc[g].abs() >= thr).collect();
        round += 1;
        if round >= 3 || de.len() >= 10 {
            break;
        }
        thr *= 0.8;
    }
    res.n_de = de.len();
    if de.is_empty() {
        res.status = "no_de";
        return res;
    }
    let (n1, n2) = (np as f64, nc as f64);
    let nt = n1 + n2;
    let mut buf = vec![0f32; np];
    let mut keyed: Vec<(f64, f64, usize)> = de
        .iter()
        .map(|&g| {
            buf.copy_from_slice(&pm[g * np..(g + 1) * np]);
            buf.sort_unstable_by(f32::total_cmp);
            let cs = &c.sorted[g * nc..(g + 1) * nc];
            let mut u = 0f64;
            let mut tie = c.tie[g];
            let mut i = 0;
            while i < np {
                let v = buf[i];
                let mut j = i + 1;
                while j < np && buf[j] == v {
                    j += 1;
                }
                let k = (j - i) as f64;
                let lb = cs.partition_point(|&x| x < v);
                let ub = cs.partition_point(|&x| x <= v);
                let e = (ub - lb) as f64;
                // each pert value: ctrl below + half of ctrl ties (pert-pert ties cancel in U)
                u += k * (lb as f64 + 0.5 * e);
                let t = e + k;
                tie += (t * t * t - t) - (e * e * e - e);
                i = j;
            }
            let var = n1 * n2 / 12.0 * ((nt + 1.0) - tie / (nt * (nt - 1.0)));
            let d = u - n1 * n2 / 2.0;
            let z = if var > 0.0 { (d - 0.5 * d.signum()) / var.sqrt() } else { 0.0 };
            (pkey(z), lfc[g], g)
        })
        .collect();
    // order(p_val, -avg_log2FC): smaller p first = larger key
    keyed.sort_by(|a, b| b.0.total_cmp(&a.0).then(b.1.total_cmp(&a.1)));
    keyed.truncate(500);
    let tg: Vec<usize> = keyed
        .iter()
        .map(|k| k.2)
        .filter(|&g| (rnz[nm.rawcol[g] as usize] + c.rawnz[g]) as f64 >= 0.01 * nt)
        .collect();
    res.n_target = tg.len();
    res.targets = tg.clone();
    if tg.is_empty() {
        res.status = "no_target";
        return res;
    }
    // clip + ridge
    let lam = 0.01;
    let det = (nt + lam) * (n1 + lam) - n1 * n1;
    let mut q = vec![0f32; tg.len()];
    let mut bp = vec![0f64; tg.len()];
    let mut bn = vec![0f64; tg.len()];
    for (t, &g) in tg.iter().enumerate() {
        buf.copy_from_slice(&pm[g * np..(g + 1) * np]);
        buf.sort_unstable_by(f32::total_cmp);
        let cs = &c.sorted[g * nc..(g + 1) * nc];
        let n = np + nc;
        let h = (n - 1) as f64 * 0.95;
        let lo = h.floor() as usize;
        let hi = (lo + 1).min(n - 1);
        // k-th smallest of merged sample, walking from the top
        let kth = |k: usize| -> f32 {
            let (mut i, mut j) = (np, nc);
            let mut v = 0.0;
            for _ in 0..(n - k) {
                if i > 0 && (j == 0 || buf[i - 1] >= cs[j - 1]) {
                    i -= 1;
                    v = buf[i];
                } else {
                    j -= 1;
                    v = cs[j];
                }
            }
            v
        };
        let (vl, vh) = (kth(lo), kth(hi));
        let qt = vl + (h - lo as f64) as f32 * (vh - vl);
        q[t] = qt;
        let sp: f64 = buf.iter().map(|&v| v.min(qt) as f64).sum();
        let mut sc = c.sum[g];
        for &v in cs.iter().rev() {
            if v <= qt {
                break;
            }
            sc -= (v - qt) as f64;
        }
        let sa = sp + sc;
        bn[t] = ((n1 + lam) * sa - n1 * sp) / det;
        bp[t] = ((nt + lam) * sp - n1 * sa) / det;
    }
    let nb2: f64 = bp.iter().map(|v| v * v).sum();
    let kk: f64 = bp.iter().zip(&bn).map(|(a, b)| a * b).sum();
    let mut sp_ = vec![0f64; np];
    let mut sc_ = vec![0f64; nc];
    for (t, &g) in tg.iter().enumerate() {
        let (qt, bt) = (q[t], bp[t]);
        for (i, &v) in pm[g * np..(g + 1) * np].iter().enumerate() {
            sp_[i] += bt * v.min(qt) as f64;
        }
        for (i, &v) in c.gm[g * nc..(g + 1) * nc].iter().enumerate() {
            sc_[i] += bt * v.min(qt) as f64;
        }
    }
    let f = |s: f64| ((s - kk) / nb2).clamp(0.0, 3.0) / 3.0;
    let xp: Vec<f64> = sp_.iter().map(|&s| f(s)).collect();
    let mut mx = xp.iter().cloned().fold(0.0, f64::max);
    for &s in &sc_ {
        mx = mx.max(f(s));
    }
    if mx < 0.01 {
        mx = 1.0;
    }
    let ps: Vec<f64> = xp.iter().map(|v| v / mx).collect();
    // dose axis
    let Some(o) = nm.genes.iter().position(|g| g == name) else {
        res.status = "gene_not_measured";
        return res;
    };
    let e = &pm[o * np..(o + 1) * np];
    let mn = e.iter().cloned().fold(c.minv[o], f32::min);
    let med = c.median[o];
    if med <= mn {
        res.status = "ctrl_median_le_min";
        return res;
    }
    let xs: Vec<f64> = e.iter().map(|&v| ((v - mn) / (med - mn)) as f64).collect();
    for i in 0..np {
        res.cells.push((cells[i], xs[i] as f32, ps[i] as f32));
    }
    res.bins = make_bins(&xs, &ps, 10, mode);
    let bx: Vec<f64> = res.bins.iter().map(|b| b.0).collect();
    let by: Vec<f64> = res.bins.iter().map(|b| b.1).collect();
    res.ed = hill::fit_ll4(&bx, &by, None);
    if res.ed.is_none() {
        res.status = "fit_failed";
    }
    res
}

pub fn run(h5ad: &str, meta: &str, raw_dir: &str, out: &str, min_cells: usize, only: Option<Vec<String>>, half: Option<u32>, mode: BinMode) -> Result<()> {
    let t0 = Instant::now();
    let nm = Norm::open(h5ad, meta)?;
    let raw = Cache::open(raw_dir)?;
    let ctrl = build_ctrl(&nm, &raw, nm.cells(0));
    eprintln!("[psz] {} controls x {} genes ready {:.1}s", ctrl.n, nm.ng, t0.elapsed().as_secs_f64());
    let mut todo: Vec<usize> = (1..nm.groups.len()).collect();
    if let Some(o) = &only {
        todo.retain(|&g| o.contains(&nm.groups[g].0));
    }
    let t1 = Instant::now();
    let res: Vec<Res> = todo
        .par_iter()
        .map(|&g| {
            let all = nm.cells(g);
            let cells: Vec<u32> = match half {
                None => all.to_vec(),
                Some(h) => all
                    .iter()
                    .cloned()
                    .filter(|&r| {
                        let mut z = (r as u64).wrapping_add(0x9E3779B97F4A7C15);
                        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                        ((z ^ (z >> 31)) & 1) as u32 == h
                    })
                    .collect(),
            };
            let mc = if half.is_some() { if all.len() >= min_cells { 20 } else { usize::MAX } } else { min_cells };
            one(&nm, &raw, &ctrl, &nm.groups[g].0, &cells, mc, mode)
        })
        .collect();
    eprintln!("[psz] {} perturbations in {:.1}s", res.len(), t1.elapsed().as_secs_f64());
    let mut w = BufWriter::new(File::create(format!("{out}.genes.tsv"))?);
    writeln!(w, "gene\tn_cells\tn_de\tn_target\tn_bins\tb\tc\td\ted50\trss\thill0\thill1\tdelta\tclass\tstatus")?;
    let mut wc = BufWriter::new(File::create(format!("{out}.cells.tsv"))?);
    writeln!(wc, "gene\tcell_row\tx\tps")?;
    let mut wt = BufWriter::new(File::create(format!("{out}.targets.tsv"))?);
    let (mut nk, mut nb) = (0, 0);
    for r in &res {
        let (fs, h0, h1, cl) = match &r.ed {
            Some(f) => {
                let (h0, h1) = (hill::predict(f, 0.0), hill::predict(f, 1.0));
                let cl = if h0 - h1 < 0.2 { "NA" } else if f.e <= 0.5 { "buffered" } else { "sensitive" };
                if cl != "NA" {
                    nk += 1;
                    if cl == "buffered" {
                        nb += 1;
                    }
                }
                (format!("{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6e}", f.b, f.c, f.d, f.e, f.rss), h0, h1, cl)
            }
            None => ("NA\tNA\tNA\tNA\tNA".into(), f64::NAN, f64::NAN, "NA"),
        };
        writeln!(w, "{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{}\t{}", r.name, r.n_cells, r.n_de, r.n_target, r.bins.len(), fs, h0, h1, h0 - h1, cl, r.status)?;
        for &(cell, x, p) in &r.cells {
            writeln!(wc, "{}\t{}\t{:.5}\t{:.5}", r.name, cell, x, p)?;
        }
        let names: Vec<&str> = r.targets.iter().map(|&g| nm.genes[g].as_str()).collect();
        writeln!(wt, "{}\t{}", r.name, names.join(","))?;
    }
    eprintln!("[psz] kept {nk}, buffered {nb}, sensitive {}; total {:.1}s", nk - nb, t0.elapsed().as_secs_f64());
    Ok(())
}
