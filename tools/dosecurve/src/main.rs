mod cache;
mod hill;
mod joint;
mod ps;
mod psz;

use anyhow::{Result, bail};

const USAGE: &str = "dosecurve cache <in.h5ad> <out_dir> [--col perturbation] [--control control]
dosecurve ps <cache_dir> <out_prefix> [--min-cells 100] [--lfc 0.1] [--no-perturb-exp]
             [--bins rank|cut] [--axis lin|log|z] [--half 0|1] [--null N] [--only G1,G2] [--cells] [--threads N]
dosecurve joint|mixscale <cache_dir> <out_prefix> [--half 0|1] [--null N] [--min-cells 100] [--outer 0] [--only ..] [--cells]";

fn opt(args: &[String], k: &str) -> Option<String> {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1).cloned())
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        bail!("{USAGE}");
    }
    if let Some(t) = opt(&a, "--threads") {
        rayon::ThreadPoolBuilder::new().num_threads(t.parse()?).build_global()?;
    }
    match a[1].as_str() {
        "cache" => cache::build(
            &a[2],
            &a[3],
            &opt(&a, "--col").unwrap_or("perturbation".into()),
            &opt(&a, "--control").unwrap_or("control".into()),
        ),
        "ps" => {
            let prm = ps::Params {
                min_cells: opt(&a, "--min-cells").map_or(Ok(100), |v| v.parse())?,
                lfc: opt(&a, "--lfc").map_or(Ok(0.1), |v| v.parse())?,
                min_pct: 0.01,
                max_targets: 500,
                min_targets: 10,
                lambda: 0.01,
                scale_factor: 3.0,
                use_perturb_exp: !a.iter().any(|x| x == "--no-perturb-exp"),
                bins: 10,
                bin_mode: match opt(&a, "--bins").as_deref() {
                    Some("rank") => ps::BinMode::Rank,
                    _ => ps::BinMode::CutUnique,
                },
                null: opt(&a, "--null").map_or(Ok(0), |v| v.parse())?,
                half: opt(&a, "--half").map(|v| v.parse()).transpose()?,
                axis: match opt(&a, "--axis").as_deref() {
                    Some("log") => ps::Axis::Log,
                    Some("z") => ps::Axis::Z,
                    _ => ps::Axis::Lin,
                },
            };
            let only = opt(&a, "--only").map(|s| s.split(',').map(String::from).collect());
            ps::run(&a[2], &a[3], &prm, only, a.iter().any(|x| x == "--cells"))
        }
        "psz" => {
            // psz <norm.h5ad> <meta_dir> <raw_cache_dir> <out_prefix>
            let only = opt(&a, "--only").map(|s| s.split(',').map(String::from).collect());
            let mode = if opt(&a, "--bins").as_deref() == Some("rank") { ps::BinMode::Rank } else { ps::BinMode::CutUnique };
            psz::run(&a[2], &a[3], &a[4], &a[5], opt(&a, "--min-cells").map_or(Ok(100), |v| v.parse())?, only,
                     opt(&a, "--half").map(|v| v.parse()).transpose()?, mode)
        }
        "joint" | "mixscale" => {
            // joint <cache_dir> <out_prefix> [--half 0|1] [--null N] [--min-cells 100] [--outer 0] [--cells]
            let prm = joint::JParams {
                method: if a[1] == "joint" { joint::Method::Joint } else { joint::Method::Mixscale },
                min_cells: opt(&a, "--min-cells").map_or(Ok(100), |v| v.parse())?,
                half: opt(&a, "--half").map(|v| v.parse()).transpose()?,
                null: opt(&a, "--null").map_or(Ok(0), |v| v.parse())?,
                min_det: opt(&a, "--min-det").map_or(Ok(0.05), |v| v.parse())?,
                n_ctrl_var: 2000,
                outer: opt(&a, "--outer").map_or(Ok(0), |v| v.parse())?,
                write_cells: a.iter().any(|x| x == "--cells"),
                no_target: a.iter().any(|x| x == "--no-target"),
                sim: opt(&a, "--sim").map(|v| {
                    // --sim GAMMA:KDIST[:EFF]
                    let f: Vec<&str> = v.split(':').collect();
                    joint::Sim {
                        gamma: f[0].parse().unwrap(),
                        kdist: f.get(1).and_then(|x| x.chars().next()).unwrap_or('u'),
                        eff: f.get(2).map_or(0.4, |x| x.parse().unwrap()),
                        n_genes: 300,
                        escape: 0.1,
                        pure: f.get(3).and_then(|x| x.chars().next()).unwrap_or(' '),
                        tau: f.get(4).map_or(0.0, |x| x.parse().unwrap()),
                    }
                }),
            };
            let only = opt(&a, "--only").map(|s| s.split(',').map(String::from).collect());
            joint::run(&a[2], &a[3], &prm, only)
        }
        _ => bail!("{USAGE}"),
    }
}
