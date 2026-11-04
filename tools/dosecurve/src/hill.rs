//! Four-parameter log-logistic (drc LL.4) least-squares fit:
//!   f(x) = c + (d - c) / (1 + exp(b (ln x - ln e)))
//! Parameters are (b, c, d, le = ln e). Levenberg-Marquardt, multi-start.

#[derive(Clone, Copy, Debug)]
pub struct Fit {
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub e: f64,
    pub rss: f64,
}

#[inline]
fn eval(p: &[f64; 4], x: f64) -> (f64, [f64; 4]) {
    let (b, c, d, le) = (p[0], p[1], p[2], p[3]);
    if x <= 0.0 {
        // limit x -> 0: ln x -> -inf
        let g = if b > 0.0 { 1.0 } else if b < 0.0 { 0.0 } else { 0.5 };
        return (c + (d - c) * g, [0.0, 1.0 - g, g, 0.0]);
    }
    let t = (b * (x.ln() - le)).clamp(-60.0, 60.0);
    let u = t.exp();
    let g = 1.0 / (1.0 + u);
    let f = c + (d - c) * g;
    let dg_du = -g * g;
    let lx = x.ln() - le;
    (f, [(d - c) * dg_du * u * lx, 1.0 - g, g, (d - c) * dg_du * u * (-b)])
}

pub fn predict(fit: &Fit, x: f64) -> f64 {
    eval(&[fit.b, fit.c, fit.d, fit.e.ln()], x).0
}

fn rss(p: &[f64; 4], xs: &[f64], ys: &[f64]) -> f64 {
    xs.iter().zip(ys).map(|(&x, &y)| (eval(p, x).0 - y).powi(2)).sum()
}

fn solve4(a: &[[f64; 4]; 4], r: &[f64; 4]) -> Option<[f64; 4]> {
    let mut m = [[0f64; 5]; 4];
    for i in 0..4 {
        m[i][..4].copy_from_slice(&a[i]);
        m[i][4] = r[i];
    }
    for col in 0..4 {
        let piv = (col..4).max_by(|&i, &j| m[i][col].abs().total_cmp(&m[j][col].abs()))?;
        if m[piv][col].abs() < 1e-300 {
            return None;
        }
        m.swap(col, piv);
        for i in 0..4 {
            if i != col {
                let f = m[i][col] / m[col][col];
                for k in col..5 {
                    m[i][k] -= f * m[col][k];
                }
            }
        }
    }
    Some([m[0][4] / m[0][0], m[1][4] / m[1][1], m[2][4] / m[2][2], m[3][4] / m[3][3]])
}

fn lm(mut p: [f64; 4], xs: &[f64], ys: &[f64], bd: Option<(f64, f64)>) -> ([f64; 4], f64) {
    if let Some((lo, hi)) = bd {
        p[1] = p[1].clamp(lo, hi);
        p[2] = p[2].clamp(lo, hi);
    }
    let mut cur = rss(&p, xs, ys);
    let mut lambda = 1e-3;
    for _ in 0..500 {
        let mut jtj = [[0f64; 4]; 4];
        let mut jtr = [0f64; 4];
        for (&x, &y) in xs.iter().zip(ys) {
            let (f, j) = eval(&p, x);
            let r = y - f;
            for a in 0..4 {
                jtr[a] += j[a] * r;
                for b in 0..4 {
                    jtj[a][b] += j[a] * j[b];
                }
            }
        }
        let mut improved = false;
        for _ in 0..30 {
            let mut a = jtj;
            for k in 0..4 {
                a[k][k] += lambda * (jtj[k][k].max(1e-12));
            }
            let Some(step) = solve4(&a, &jtr) else {
                lambda *= 10.0;
                continue;
            };
            let mut np = p;
            for k in 0..4 {
                np[k] += step[k];
            }
            np[3] = np[3].clamp(-30.0, 30.0);
            if let Some((lo, hi)) = bd {
                np[1] = np[1].clamp(lo, hi);
                np[2] = np[2].clamp(lo, hi);
            }
            let nr = rss(&np, xs, ys);
            if nr.is_finite() && nr < cur {
                let rel = (cur - nr) / cur.max(1e-300);
                p = np;
                cur = nr;
                lambda = (lambda * 0.3).max(1e-12);
                improved = true;
                if rel < 1e-12 {
                    return (p, cur);
                }
                break;
            }
            lambda *= 10.0;
        }
        if !improved {
            break;
        }
    }
    (p, cur)
}

/// Best LL.4 fit over a grid of starting values. Needs >= 4 points.
/// `bd` bounds the asymptotes c and d (projected LM).
pub fn fit_ll4(xs: &[f64], ys: &[f64], bd: Option<(f64, f64)>) -> Option<Fit> {
    if xs.len() < 4 {
        return None;
    }
    let ymin = ys.iter().cloned().fold(f64::INFINITY, f64::min);
    let ymax = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut pos: Vec<f64> = xs.iter().cloned().filter(|&x| x > 0.0).collect();
    pos.sort_by(f64::total_cmp);
    if pos.is_empty() {
        return None;
    }
    let mut les = vec![];
    for q in [0.1, 0.25, 0.5, 0.75, 0.9] {
        les.push(pos[((pos.len() - 1) as f64 * q).round() as usize].ln());
    }
    les.push(0.5f64.ln());
    let mut best: Option<([f64; 4], f64)> = None;
    for &b in &[-8.0, -3.0, -1.0, -0.3, 0.3, 1.0, 3.0, 8.0] {
        for &le in &les {
            // decreasing curve in x when b > 0 means f(0) = d
            for &(c, d) in &[(ymin, ymax), (ymax, ymin)] {
                let (p, r) = lm([b, c, d, le], xs, ys, bd);
                if r.is_finite() && best.map_or(true, |(_, br)| r < br - 1e-15) {
                    best = Some((p, r));
                }
            }
        }
    }
    best.map(|(p, r)| Fit { b: p[0], c: p[1], d: p[2], e: p[3].exp(), rss: r })
}
