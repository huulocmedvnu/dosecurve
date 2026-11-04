//! Convert an AnnData h5ad (CSR or dense raw counts) into a compact sparse
//! cache with perturbation groups. Layout (directory):
//!   genes.txt   gene names, one per line
//!   groups.tsv  name, offset into members.bin, n cells (control first)
//!   members.bin u32 cell rows of each group
//!   indptr.bin  u64 x (n_cells + 1), row offsets (original row order)
//!   lib.bin     f32 x n_cells, total counts per cell
//!   idx.bin     u16 x nnz, gene index
//!   cnt.bin     u16 x nnz, UMI count
//! The h5ad is decompressed once; later passes mmap these files.

use anyhow::{Context, Result, bail};
use hdf5_metno as hdf5;
use hdf5::types::VarLenUnicode;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

fn read_strings(ds: &hdf5::Dataset) -> Result<Vec<String>> {
    let v: Vec<VarLenUnicode> = ds.read_raw()?;
    Ok(v.into_iter().map(|s| s.as_str().to_string()).collect())
}

fn read_i64(ds: &hdf5::Dataset) -> Result<Vec<i64>> {
    let dt = ds.dtype()?;
    Ok(match dt.size() {
        1 => ds.read_raw::<i8>()?.into_iter().map(|v| v as i64).collect(),
        2 => ds.read_raw::<i16>()?.into_iter().map(|v| v as i64).collect(),
        4 => ds.read_raw::<i32>()?.into_iter().map(|v| v as i64).collect(),
        _ => ds.read_raw::<i64>()?,
    })
}

/// Labels of an obs column stored either as categorical group or plain strings.
pub fn read_obs_column(f: &hdf5::File, col: &str) -> Result<Vec<String>> {
    let path = format!("obs/{col}");
    if let Ok(g) = f.group(&path) {
        let cd = g.dataset("categories")?;
        let cats = match read_strings(&cd) {
            Ok(v) => v,
            Err(_) => read_i64(&cd)?.into_iter().map(|x| x.to_string()).collect(),
        };
        let codes = read_i64(&g.dataset("codes")?)?;
        let cats: Vec<String> = cats;
        return Ok(codes
            .into_iter()
            .map(|c| if c < 0 { "NA".to_string() } else { cats[c as usize].clone() })
            .collect());
    }
    let ds = f.dataset(&path).with_context(|| format!("obs column {col} not found"))?;
    if let Ok(v) = read_strings(&ds) {
        return Ok(v);
    }
    if let Ok(v) = read_i64(&ds) {
        return Ok(v.into_iter().map(|x| x.to_string()).collect());
    }
    let v: Vec<f64> = ds.read_raw()?;
    Ok(v.into_iter().map(|x| x.to_string()).collect())
}



fn var_names(f: &hdf5::File) -> Result<Vec<String>> {
    let var = f.group("var")?;
    let col: String = match var.attr("_index") {
        Ok(a) => a.read_scalar::<VarLenUnicode>()?.as_str().to_string(),
        Err(_) => "_index".into(),
    };
    read_strings(&var.dataset(&col)?)
}

struct Sink {
    idx: BufWriter<File>,
    cnt: BufWriter<File>,
    indptr: Vec<u64>,
    lib: Vec<f32>,
    nonint: u64,
}

impl Sink {
    #[inline]
    fn push_row(&mut self, vals: impl Iterator<Item = (u32, f32)>) -> Result<()> {
        let mut tot = 0f64;
        let mut n = 0u64;
        for (g, v) in vals {
            if v == 0.0 {
                continue;
            }
            let rv = v.round();
            if (v - rv).abs() > 1e-4 || rv < 0.0 || rv > 65535.0 {
                self.nonint += 1;
            }
            self.idx.write_all(&(g as u16).to_le_bytes())?;
            self.cnt.write_all(&(rv as u16).to_le_bytes())?;
            tot += v as f64;
            n += 1;
        }
        let last = *self.indptr.last().unwrap();
        self.indptr.push(last + n);
        self.lib.push(tot as f32);
        Ok(())
    }
}

pub fn build(h5ad: &str, out: &str, col: &str, control: &str) -> Result<()> {
    let t0 = Instant::now();
    let out = Path::new(out);
    std::fs::create_dir_all(out)?;
    let f = hdf5::File::open(h5ad)?;
    let genes = var_names(&f)?;
    let n_genes = genes.len();
    if n_genes > u16::MAX as usize {
        bail!("{n_genes} genes do not fit u16 indices");
    }
    let mut gf = BufWriter::new(File::create(out.join("genes.txt"))?);
    for g in &genes {
        writeln!(gf, "{g}")?;
    }
    gf.flush()?;

    // groups: control first, then sorted names; members in original row order
    let labels = read_obs_column(&f, col)?;
    let n_cells = labels.len();
    let mut names: Vec<String> = labels.clone();
    names.sort();
    names.dedup();
    let Some(p) = names.iter().position(|n| n == control) else {
        bail!("control label '{control}' not in obs/{col}");
    };
    let c = names.remove(p);
    names.insert(0, c);
    let gid: std::collections::HashMap<&str, usize> =
        names.iter().enumerate().map(|(i, n)| (n.as_str(), i)).collect();
    let mut members: Vec<Vec<u32>> = vec![vec![]; names.len()];
    for (r, l) in labels.iter().enumerate() {
        members[gid[l.as_str()]].push(r as u32);
    }
    let mut gfile = BufWriter::new(File::create(out.join("groups.tsv"))?);
    let mut flat: Vec<u32> = Vec::with_capacity(n_cells);
    for (i, n) in names.iter().enumerate() {
        writeln!(gfile, "{n}\t{}\t{}", flat.len(), members[i].len())?;
        flat.extend_from_slice(&members[i]);
    }
    gfile.flush()?;
    // batch (gem group) per cell, for Replogle-style z normalisation
    for bc in ["gemgroup", "batch", "gem_group"] {
        if let Ok(b) = read_obs_column(&f, bc) {
            let mut u: Vec<&String> = b.iter().collect();
            u.sort();
            u.dedup();
            let id: std::collections::HashMap<&String, u16> = u.iter().enumerate().map(|(i, s)| (*s, i as u16)).collect();
            let v: Vec<u16> = b.iter().map(|s| id[s]).collect();
            File::create(out.join("batch.bin"))?.write_all(bytemuck::cast_slice(&v))?;
            eprintln!("[cache] batch column '{bc}': {} levels", u.len());
            break;
        }
    }
    File::create(out.join("members.bin"))?.write_all(bytemuck::cast_slice(&flat))?;

    let mut sink = Sink {
        idx: BufWriter::with_capacity(8 << 20, File::create(out.join("idx.bin"))?),
        cnt: BufWriter::with_capacity(8 << 20, File::create(out.join("cnt.bin"))?),
        indptr: Vec::with_capacity(n_cells + 1),
        lib: Vec::with_capacity(n_cells),
        nonint: 0,
    };
    sink.indptr.push(0);
    let (mut t_read, mut t_conv) = (0f64, 0f64);

    if let Ok(x) = f.group("X") {
        // CSR
        let indptr = read_i64(&x.dataset("indptr")?)?;
        let data_ds = x.dataset("data")?;
        let ind_ds = x.dataset("indices")?;
        let block = 64usize << 20;
        let mut r0 = 0usize;
        while r0 < n_cells {
            let a = indptr[r0] as usize;
            let mut r1 = r0 + 1;
            while r1 < n_cells && (indptr[r1 + 1] as usize) - a <= block {
                r1 += 1;
            }
            let b = indptr[r1] as usize;
            let tr = Instant::now();
            let d: Vec<f32> = data_ds.read_slice_1d(a..b)?.to_vec();
            let ii: Vec<i32> = ind_ds.read_slice_1d(a..b)?.to_vec();
            t_read += tr.elapsed().as_secs_f64();
            let ts = Instant::now();
            for r in r0..r1 {
                let (s, e) = ((indptr[r] as usize) - a, (indptr[r + 1] as usize) - a);
                sink.push_row((s..e).map(|k| (ii[k] as u32, d[k])))?;
            }
            t_conv += ts.elapsed().as_secs_f64();
            r0 = r1;
        }
    } else {
        // dense, read bands aligned to the chunk rows
        let x = f.dataset("X")?;
        let shape = x.shape();
        if shape[0] != n_cells || shape[1] != n_genes {
            bail!("X shape {:?} vs obs/var {n_cells}x{n_genes}", shape);
        }
        let crow = x.chunk().map(|c| c[0]).unwrap_or(1024).max(1);
        let band = crow * (4096 / crow).max(1);
        eprintln!("[cache] dense X {:?}, chunk {:?}, band {band} rows", shape, x.chunk());
        let mut r0 = 0usize;
        while r0 < n_cells {
            let r1 = (r0 + band).min(n_cells);
            let tr = Instant::now();
            let m: ndarray_like::Band = ndarray_like::read_band(&x, r0, r1, n_genes)?;
            t_read += tr.elapsed().as_secs_f64();
            let ts = Instant::now();
            for i in 0..(r1 - r0) {
                let row = &m.data[i * n_genes..(i + 1) * n_genes];
                sink.push_row(row.iter().enumerate().map(|(g, &v)| (g as u32, v)))?;
            }
            t_conv += ts.elapsed().as_secs_f64();
            if (r0 / band) % 10 == 0 {
                eprintln!("[cache] rows {r1}/{n_cells}  read {t_read:.1}s convert {t_conv:.1}s");
            }
            r0 = r1;
        }
    }
    if sink.nonint > 0 {
        bail!("{} values are not integer counts in [0, 65535]", sink.nonint);
    }
    sink.idx.flush()?;
    sink.cnt.flush()?;
    File::create(out.join("indptr.bin"))?.write_all(bytemuck::cast_slice(&sink.indptr))?;
    File::create(out.join("lib.bin"))?.write_all(bytemuck::cast_slice(&sink.lib))?;
    eprintln!(
        "[cache] {n_cells} cells x {n_genes} genes, nnz {}, {} groups, {:.1}s (h5 read {t_read:.1}s, convert {t_conv:.1}s)",
        sink.indptr[n_cells],
        names.len(),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

mod ndarray_like {
    use super::*;
    pub struct Band {
        pub data: Vec<f32>,
    }
    pub fn read_band(x: &hdf5::Dataset, r0: usize, r1: usize, ng: usize) -> Result<Band> {
        let a: hdf5::Result<_> = x.read_slice_2d::<f32, _>((r0..r1, 0..ng));
        let a = a?;
        let data = a.into_raw_vec_and_offset().0;
        Ok(Band { data })
    }
}
