"""Manuscript figures 1 to 4 (plotly, plotly_white, Arial, black text, no gridlines).

usage: python -I tools/figures.py
Inputs: out/bench_v4/*_metrics.tsv and *_full.log, out/simgrid_eff_v4/fpr.tsv,
out/figures/data/*_test_pred.tsv and *_regime.tsv, out/gtest/summary.tsv,
out/final/fig3_map.tsv, out/real/target_mix.tsv, out/rpe1/target_mix.tsv,
out/final/crossline_*.tsv, out/figures/data/cells/d_*.tsv,
out/figures/data/k562_target_mix_examples.tsv, out/figures/fig1a.svg.
Writes out/figures/fig{1..4}.html, .pdf, .png and the plotted values as
out/figures/data/fig*_values.tsv.
"""
import base64, os, re, subprocess
import numpy as np, pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.special import gammaln
from scipy.stats import fisher_exact
from sklearn.metrics import roc_curve, roc_auc_score

O = "out/figures"
D = f"{O}/data"
INK = "#111111"
MC = {"Song PS": "#636EFA", "Mixscale": "#EF553B", "Joint EM": "#00CC96"}
CC = {"buffered": "#AB63FA", "sensitive": "#E3732B", "not classified": "#D3D3D3"}
LC = {"K562": "#1F9BB5", "RPE1": "#E0457B"}
NL = ["buffered", "sensitive"]
W = 900


def style(fig, h):
    fig.update_layout(template="plotly_white", width=W, height=h, font=dict(family="Arial", size=13, color=INK),
                      margin=dict(l=70, r=20, t=50, b=60), legend=dict(font=dict(color=INK)))
    fig.update_xaxes(showgrid=False, zeroline=False, showline=True, linecolor=INK, ticks="outside", tickcolor=INK,
                     tickfont=dict(color=INK), title_font=dict(color=INK))
    fig.update_yaxes(showgrid=False, zeroline=False, showline=True, linecolor=INK, ticks="outside", tickcolor=INK,
                     tickfont=dict(color=INK), title_font=dict(color=INK))
    seen = set()
    for a in fig.layout.annotations:
        sz = a.font.size if a.font is not None and a.font.size is not None else 14
        a.font = dict(family="Arial", size=14 if sz >= 16 else sz, color=INK)
        m = re.match(r"^([a-z])  (.*)$", a.text or "")
        if m:
            # panel letter only on the first subplot of each group, in bold
            a.text = m.group(2) if m.group(1) in seen else f"<b>{m.group(1)}</b>  {m.group(2)}"
            seen.add(m.group(1))
    return fig


def save(fig, name):
    fig.write_html(f"{O}/{name}.html", include_plotlyjs="cdn")
    fig.write_image(f"{O}/{name}.pdf")
    fig.write_image(f"{O}/{name}.png", scale=3)


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def cls(x):
    return x.where(x.isin(NL), "not classified")


def peak(log):
    t = open(log).read()
    real = float(re.search(r"([\d.]+) real", t).group(1))
    mem = float(re.search(r"(\d+)\s+peak memory footprint", t).group(1)) / 1e9
    return real, mem


# ---------------------------------------------------------------- figure 1
def fig1():
    svg = f"{O}/fig1a.svg"
    png = f"{O}/fig1a.png"
    subprocess.run(["rsvg-convert", "-z", "3", "-o", png, svg], check=True)
    b64 = base64.b64encode(open(png, "rb").read()).decode()
    bm = {l: pd.read_csv(f"out/bench_v4/{l}_metrics.tsv", sep="\t", index_col=0) for l in ["ess", "rpe1"]}
    meth = list(MC)
    fig = make_subplots(rows=3, cols=12, row_heights=[0.36, 0.32, 0.32], vertical_spacing=0.1, horizontal_spacing=0.035,
                        specs=[[{"colspan": 12}] + [None] * 11,
                               [{"colspan": 4}, None, None, None, {"colspan": 4}, None, None, None, {"colspan": 4}, None, None, None],
                               [{"colspan": 3}, None, None, {"colspan": 3}, None, None, {"colspan": 3}, None, None, {"colspan": 3}, None, None]],
                        subplot_titles=["a  Two-step scoring and the joint model", "b  Mock groups with p < 0.05 (fraction)",
                                        "b  DE genes per mock group (median)", "b  Mock groups given a class (fraction)",
                                        "c  Halves, kappa", "c  Halves, dose Spearman", "c  Wall clock (s)", "c  Peak memory (GB)"])
    fig.add_layout_image(dict(source="data:image/png;base64," + b64, xref="x domain", yref="y domain", x=0, y=1,
                              sizex=1, sizey=1, xanchor="left", yanchor="top", sizing="contain", layer="below"), row=1, col=1)
    fig.update_xaxes(visible=False, row=1, col=1)
    fig.update_yaxes(visible=False, row=1, col=1)
    vals = []
    LINES = [("ess", "K562"), ("rpe1", "RPE1")]

    def bars(row, col, y_of, fmt, ttl, panel, yr=None):
        for li, (lab, sym) in enumerate(LINES):
            y = y_of(lab)
            fig.add_trace(go.Bar(x=meth, y=y, marker_color=[MC[m] for m in meth], marker_line_width=0, offsetgroup=li,
                                 showlegend=False, text=[fmt.format(v) for v in y], textposition="outside",
                                 textfont=dict(color=INK, size=10), marker_pattern_shape="" if li == 0 else "/",
                                 hovertemplate=f"{sym} %{{x}}: %{{y}}<extra></extra>"), row=row, col=col)
            vals.extend(dict(panel=panel, line=sym, method=m, value=v) for m, v in zip(meth, y))
        if yr:
            fig.update_yaxes(range=yr, row=row, col=col)

    bars(2, 1, lambda l: [bm[l].loc["mock_p05"].iloc[i] for i in range(3)], "{:.3f}", "fraction of mock groups", "1b p05", [0, 1.15])
    bars(2, 5, lambda l: [bm[l].loc["mock_de_median"].iloc[i] for i in range(3)], "{:.0f}", "median DE genes per group", "1b DE", [0, 1050])
    bars(2, 9, lambda l: [bm[l].loc["mock_classified"].iloc[i] for i in range(3)], "{:.3f}", "fraction of mock groups", "1b classified", [0, 0.04])
    bars(3, 1, lambda l: [bm[l].loc["half_kappa_3state"].iloc[i] for i in range(3)], "{:.2f}", "Cohen kappa, 3 classes", "1c kappa", [0, 1])
    bars(3, 4, lambda l: [bm[l].loc["half_spearman_dose"].iloc[i] for i in range(3)], "{:.2f}", "Spearman, dose parameter", "1c spearman", [0, 1])
    bars(3, 7, lambda l: [peak(f"out/bench_v4/{l}_{m}_full.log")[0] for m in ["song", "mix", "joint"]], "{:.0f}", "seconds", "1c seconds", [0, 150])
    bars(3, 10, lambda l: [peak(f"out/bench_v4/{l}_{m}_full.log")[1] for m in ["song", "mix", "joint"]], "{:.2f}", "GB", "1c GB", [0, 1.3])
    # legend for the two lines (texture)
    for sym, pat in [("K562", ""), ("RPE1", "/")]:
        fig.add_trace(go.Bar(x=[None], y=[None], name=sym, marker_color="#888888", marker_pattern_shape=pat, showlegend=True), row=2, col=1)
    fig.update_layout(barmode="group", bargap=0.2, legend=dict(orientation="h", x=1, xanchor="right", y=0.62))
    style(fig, 1050)
    fig.update_xaxes(tickangle=45, tickfont=dict(size=10))
    save(fig, "fig1")
    pd.DataFrame(vals).to_csv(f"{D}/fig1_values.tsv", sep="\t", index=False, float_format="%.4g")


# ---------------------------------------------------------------- figure 2
DOSES = ["spread", "moder", "high", "extreme"]
DLAB = {"spread": "Beta(2,2)", "moder": "Beta(4,1.5)", "high": "Beta(15,1.5)", "extreme": "Beta(40,1)",
        "bimod0.2": "bimodal 20%", "bimod0.4": "bimodal 40%", "bimod0.6": "bimodal 60%"}


def fig2():
    f = pd.read_csv("out/simgrid_eff_v4/fpr.tsv", sep="\t")
    f = f[f.seed == 2]
    tiers = ["weak", "medium", "strong", "ultra"]
    fig = make_subplots(rows=3, cols=4, vertical_spacing=0.13, horizontal_spacing=0.035,
                        specs=[[{}, {}, {}, {}], [{"colspan": 2}, None, {"colspan": 2}, None], [{}, {}, {}, {}]],
                        subplot_titles=[f"a  {t} effect" if i == 0 else f"{t} effect" for i, t in enumerate(tiers)] +
                                       ["b  Map on held-out simulations (ROC)", "b  Real perturbations, P(safe)"] +
                                       ["c  True gamma 0.5", "true gamma 1", "true gamma 2", "c  Mapped safe, false nonlinear"])
    vals = []
    cols_all = DOSES + ["bimod0.2", "bimod0.4", "bimod0.6"]
    for ci, t in enumerate(tiers, start=1):
        x = f[f.eff == t]
        cols_t = [d for d in cols_all if (x.dose == d).any()]
        z, txt = [], []
        for tau in [0.0, 0.3, 0.6]:
            row, tr = [], []
            for d in cols_t:
                v = x[(x.dose == d) & (np.isclose(x.tau, tau))].em
                row.append(v.iloc[0] if len(v) else None)
                tr.append(f"{v.iloc[0]:.2f}" if len(v) else "")
                if len(v):
                    vals.append(dict(panel="2a", tier=t, dose=d, tau=tau, em_false=v.iloc[0]))
            z.append(row)
            txt.append(tr)
        fig.add_trace(go.Heatmap(z=z, x=[DLAB[d] for d in cols_t], y=["tau 0", "tau 0.3", "tau 0.6"], zmin=0, zmax=1,
                                 colorscale="Blues", showscale=ci == 4, xgap=2, ygap=2,
                                 colorbar=dict(title=dict(text="EM false<br>nonlinear", side="top", font=dict(size=11)), len=0.28, y=0.85, x=1.0, xanchor="left", thickness=10),
                                 hovertemplate="%{x}, %{y}: %{z:.2f}<extra></extra>"), row=1, col=ci)
        # cell values as a text layer: white on dark cells (value >= 0.40)
        tx, ty, tt, tc = [], [], [], []
        for yi, yl in enumerate(["tau 0", "tau 0.3", "tau 0.6"]):
            for xi, d in enumerate(cols_t):
                if z[yi][xi] is not None:
                    tx.append(DLAB[d]); ty.append(yl); tt.append(txt[yi][xi]); tc.append("white" if z[yi][xi] >= 0.40 else INK)
        fig.add_trace(go.Scatter(x=tx, y=ty, mode="text", text=tt, textfont=dict(size=8, color=tc), showlegend=False, hoverinfo="skip"), row=1, col=ci)
        fig.update_xaxes(tickangle=60, tickfont=dict(size=10), showline=False, ticks="", row=1, col=ci)
        fig.update_yaxes(showline=False, ticks="", showticklabels=ci == 1, autorange="reversed", row=1, col=ci)
    heat_axes = ["xaxis", "xaxis2", "xaxis3", "xaxis4", "yaxis", "yaxis2", "yaxis3", "yaxis4"]
    # ROC
    for lab in ["k562", "rpe1"]:
        p = pd.read_csv(f"{D}/{lab}_test_pred.tsv", sep="\t")
        fpr, tpr, _ = roc_curve(p.safe, p.p)
        auc = roc_auc_score(p.safe, p.p)
        nm = lab.upper()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=LC[nm], width=2), name=f"{nm} grid, AUC {auc:.2f}"), row=2, col=1)
        vals.append(dict(panel="2b ROC", line=nm, auc=auc))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#999999", width=1, dash="dot"), showlegend=False), row=2, col=1)
    fig.update_xaxes(title_text="false positive rate", range=[0, 1], row=2, col=1)
    fig.update_yaxes(title_text="true positive rate", range=[0, 1.02], row=2, col=1)
    for lab in ["k562", "rpe1"]:
        r = pd.read_csv(f"{D}/{lab}_regime.tsv", sep="\t", index_col=0)
        h, e = np.histogram(r.p_safe, bins=np.linspace(0, 1, 21))
        nm = lab.upper()
        fig.add_trace(go.Scatter(x=np.repeat(e, 2)[1:-1], y=np.repeat(h / h.sum(), 2), mode="lines", line=dict(color=LC[nm], width=2),
                                 showlegend=False, hovertemplate=f"{nm}: %{{y:.3f}}<extra></extra>"), row=2, col=3)
        vals.append(dict(panel="2b safe", line=nm, safe_frac=(r.p_safe >= 0.5).mean(), n=len(r)))
        fig.add_annotation(x=0.06, y=(0.32 if nm == "K562" else 0.28), xref="x6", yref="y6",
                           text=f"{nm}: {(r.p_safe >= 0.5).mean():.0%} safe (n {len(r)})", showarrow=False,
                           xanchor="left", font=dict(color=INK, size=11))
    fig.add_vline(x=0.5, line=dict(color="#999999", width=1, dash="dot"), row=2, col=3)
    fig.update_xaxes(title_text="P(safe)", range=[0, 1], row=2, col=3)
    fig.update_yaxes(title_text="fraction of groups", row=2, col=3)
    # gamma panels from out/gtest
    g = pd.read_csv("out/gtest/summary.tsv", sep="\t")
    keym = {"Song PS": "song", "Mixscale": "mix", "Joint EM": "v4"}
    for ci, gm in enumerate([0.5, 1.0, 2.0], start=1):
        x = g[np.isclose(g.gamma, gm)]
        for k, (lab, mk) in enumerate(keym.items()):
            for j, (c, nm) in enumerate([("S", "sensitive"), ("B", "buffered")]):
                v = x[f"{mk}_{c}"]
                fig.add_trace(go.Scatter(x=[j + (k - 1) * 0.22], y=[v.mean()], mode="markers", marker=dict(symbol="diamond", size=10, color=MC[lab]),
                                         error_y=dict(type="data", symmetric=False, array=[v.max() - v.mean()], arrayminus=[v.mean() - v.min()],
                                                      color=MC[lab], thickness=1.2, width=3),
                                         name=lab, showlegend=(ci == 1 and j == 0), legendgroup=lab,
                                         hovertemplate=f"{lab}, {nm} calls: mean %{{y:.2f}}<extra></extra>"), row=3, col=ci)
                vals.append(dict(panel="2c", gamma=gm, method=lab, call=nm, mean=v.mean(), min=v.min(), max=v.max()))
        fig.update_xaxes(tickvals=[0, 1], ticktext=["sensitive", "buffered"], range=[-0.5, 1.5], row=3, col=ci)
        fig.update_yaxes(range=[0, 0.55], title_text="fraction of groups" if ci == 1 else None, row=3, col=ci)
    m = pd.read_csv("out/final/fig3_map.tsv", sep="\t", index_col=0)
    for li, (cname, nm) in enumerate([(m.columns[0], "K562"), (m.columns[1], "RPE1")]):
        y = [m.loc["song_fpr_0.5", cname], m.loc["mix_fpr_0.5", cname], m.loc["em_fpr_0.5", cname]]
        fig.add_trace(go.Bar(x=list(MC), y=y, marker_color=list(MC.values()), offsetgroup=li, marker_pattern_shape="" if li == 0 else "/",
                             showlegend=False, text=[f"{v:.2f}" for v in y], textposition="outside", textfont=dict(size=10, color=INK),
                             hovertemplate=f"{nm} grid %{{x}}: %{{y:.3f}}<extra></extra>"), row=3, col=4)
        vals += [dict(panel="2c mapped safe", line=nm, method=k, value=v) for k, v in zip(MC, y)]
    fig.update_xaxes(tickangle=45, tickfont=dict(size=10), row=3, col=4)
    fig.update_yaxes(range=[0, 0.45], row=3, col=4)
    for nm, pat in [("K562 grid", ""), ("RPE1 grid", "/")]:
        fig.add_trace(go.Bar(x=[None], y=[None], name=nm, marker_color="#888888", marker_pattern_shape=pat), row=3, col=4)
    fig.update_layout(barmode="group", legend=dict(orientation="h", x=0, y=-0.1))
    style(fig, 1100)
    fig.update_layout(margin=dict(r=150))
    for a in heat_axes:
        fig.layout[a].showline = False
        fig.layout[a].ticks = ""
    save(fig, "fig2")
    pd.DataFrame(vals).to_csv(f"{D}/fig2_values.tsv", sep="\t", index=False, float_format="%.4g")


# ---------------------------------------------------------------- figure 3
def nb_logpmf(c, mean, r):
    mean = np.maximum(mean, 1e-12)
    return gammaln(c + r) - gammaln(r) - gammaln(c + 1) + r * np.log(r / (r + mean)) + c * np.log(mean / (r + mean))


def count_dist(row, L, cmax):
    K = np.linspace(0, 0.995, 100)
    x = np.clip(K, 1e-3, 1 - 1e-3)
    lw = (row.beta_a - 1) * np.log(x) + (row.beta_b - 1) * np.log(1 - x)
    wb = np.exp(lw - lw.max()); wb /= wb.sum()
    wg = (1 - row.pi_graded) * wb; wg[0] += row.pi_graded
    wa = np.zeros(100); wa[0] = row.pi_aon; wa[np.argmin(np.abs(K - row.k_hi))] += 1 - row.pi_aon
    cs = np.arange(cmax + 1)
    out = {}
    for nm, w in [("all-or-none", wa), ("graded", wg)]:
        P = np.exp(nb_logpmf(cs[None, :, None], row.mu * L[:, None, None] * (1 - K)[None, None, :], row.size))  # cells x c x K
        p = (P * w[None, None, :]).sum(2).mean(0)
        p[-1] = 1 - p[:-1].sum()
        out[nm] = p
    return out


def binned(d, tm, bins):
    K = np.linspace(0, 0.995, 51)
    lam = tm * (d.L.values / d.L.mean())[:, None] * (1 - K)[None, :]
    from scipy.stats import poisson
    lik = poisson.pmf(d.cnt.values[:, None], lam)
    pi = np.full(51, 1 / 51)
    for _ in range(300):
        w = lik * pi; w /= w.sum(1, keepdims=True); pi = w.mean(0)
    u = d.u.values
    res = {"observed": [], "se": []}
    for b in np.unique(bins):
        res["observed"].append(u[bins == b].mean()); res["se"].append(u[bins == b].std() / np.sqrt((bins == b).sum()))
    for gm in (1, 4):
        xg = w @ K ** gm
        mm = (xg @ u) / (xg @ xg)
        res[f"gamma {gm}"] = [mm * xg[bins == b].mean() for b in np.unique(bins)]
    return res


def fig3():
    r = pd.read_csv(f"{D}/k562_regime.tsv", sep="\t", index_col=0)
    t = pd.read_csv("out/real/target_mix.tsv", sep="\t").set_index("group")
    r = r.join(t[["tmean", "llr"]])
    s = r[r.p_safe >= 0.5].copy()
    ex = pd.read_csv(f"{D}/k562_target_mix_examples.tsv", sep="\t").set_index("group")
    genes = ["PSMC4", "POLR2B", "RPS19"]
    fig = make_subplots(rows=3, cols=3, vertical_spacing=0.13, horizontal_spacing=0.09,
                        specs=[[{"colspan": 2}, None, {}], [{}, {}, {}], [{}, {}, {}]],
                        subplot_titles=["a  K562 safe regime, class by method", "b  EM buffered by phenotype strength"] +
                                       [f"c  {genes[0]}, target UMI per cell", f"{genes[1]}, target UMI per cell", f"{genes[2]}, target UMI per cell"] +
                                       [f"c  {genes[0]}, response by target bin", f"{genes[1]}, response by target bin", f"{genes[2]}, response by target bin"])
    vals = []
    meth = {"Song PS": "song", "Mixscale": "mix", "Joint EM": "em"}
    for c in ["buffered", "sensitive", "not classified"]:
        y = [(cls(s[k]) == c).mean() for k in meth.values()]
        fig.add_trace(go.Bar(y=list(meth), x=y, orientation="h", name=c, marker_color=CC[c], marker_line=dict(color="white", width=2),
                             text=[f"{v:.0%}" for v in y], textposition="inside", insidetextanchor="middle", textfont=dict(color=INK, size=11),
                             hovertemplate=f"%{{y}}, {c}: %{{x:.3f}}<extra></extra>"), row=1, col=1)
        vals += [dict(panel="3a", method=m, cls=c, value=v, n=len(s)) for m, v in zip(meth, y)]
    fig.update_xaxes(title_text=f"fraction of safe-regime groups (n {len(s)})", range=[0, 1], row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    s["Tq"] = pd.qcut(s["T"], 3, labels=["low", "middle", "high"])
    for q, x in s.groupby("Tq", observed=True):
        k, n = (x.em == "buffered").sum(), len(x)
        lo, hi = wilson(k, n)
        fig.add_trace(go.Scatter(x=[q], y=[k / n], mode="markers", marker=dict(symbol="diamond", size=10, color=CC["buffered"]),
                                 error_y=dict(type="data", symmetric=False, array=[hi - k / n], arrayminus=[k / n - lo], color=CC["buffered"], thickness=1.2, width=3),
                                 showlegend=False, hovertemplate=f"T tertile {q} (median {x['T'].median():.1f}): %{{y:.2f}}, n {n}<extra></extra>"), row=1, col=3)
        vals.append(dict(panel="3b", tertile=q, T_median=x["T"].median(), buffered=k / n, lo=lo, hi=hi, n=n))
    fig.update_xaxes(title_text="phenotype T tertile", row=1, col=3)
    fig.update_yaxes(title_text="EM buffered", range=[0, 0.75], row=1, col=3)
    b = s[(s.em == "buffered") & (s.tmean >= 0.5)]
    o = s[(s.em != "buffered") & (s.tmean >= 0.5)]
    vals.append(dict(panel="3b target test", group="EM buffered", graded=(2 * b.llr > 3.84).mean(), n=len(b)))
    vals.append(dict(panel="3b target test", group="other safe", graded=(2 * o.llr > 3.84).mean(), n=len(o)))
    for ci, gname in enumerate(genes, start=1):
        d = pd.read_csv(f"{D}/cells/d_{gname}.tsv", sep="\t")
        row = ex.loc[gname]
        L = d.L.values
        tm = row.mu * L.mean()
        cmax = 5 if tm < 5 else int(np.ceil(np.quantile(d.cnt, 0.99)))
        if tm < 5:
            obs = np.bincount(np.minimum(d.cnt.astype(int), cmax), minlength=cmax + 1) / len(d)
            exp = count_dist(row, L, cmax)
            xs = [str(i) for i in range(cmax)] + [f">={cmax}"]
            fig.add_trace(go.Bar(x=xs, y=obs, marker_color="#BBBBBB", showlegend=False, hovertemplate="observed %{x}: %{y:.3f}<extra></extra>"), row=2, col=ci)
            for nm, dash, colr in [("all-or-none", "dot", "#555555"), ("graded", "solid", CC["buffered"])]:
                fig.add_trace(go.Scatter(x=xs, y=exp[nm], mode="lines+markers", line=dict(color=colr, width=2, dash=dash), marker=dict(size=8),
                                         name=f"{nm} fit", showlegend=ci == 1, legendgroup=nm), row=2, col=ci)
                vals += [dict(panel="3c counts", gene=gname, model=nm, count=xx, p=pp) for xx, pp in zip(xs, exp[nm])]
            vals += [dict(panel="3c counts", gene=gname, model="observed", count=xx, p=pp) for xx, pp in zip(xs, obs)]
            fig.update_xaxes(title_text="target UMI", row=2, col=ci)
            bins = np.minimum(d.cnt.values.astype(int), 3)
            blab = ["0", "1", "2", ">=3"]
        else:
            rel = d.cnt.values / (tm * L / L.mean())
            hh, ee = np.histogram(rel, bins=np.linspace(0, 1.6, 17))
            fig.add_trace(go.Bar(x=(ee[:-1] + ee[1:]) / 2, y=hh / len(d), marker_color="#BBBBBB", showlegend=False,
                                 hovertemplate="relative target %{x:.2f}: %{y:.3f}<extra></extra>"), row=2, col=ci)
            vals += [dict(panel="3c counts", gene=gname, model="observed", count=round(xx, 2), p=pp) for xx, pp in zip((ee[:-1] + ee[1:]) / 2, hh / len(d))]
            fig.update_xaxes(title_text="target UMI / control expectation", row=2, col=ci)
            q = np.quantile(rel, [0.2, 0.4, 0.6, 0.8])
            bins = np.digitize(rel, q)
            blab = ["q1", "q2", "q3", "q4", "q5"]
        fig.add_annotation(text=f"knockdown mean {row.beta_mean:.2f}, escapers {row.pi_graded:.2f}, LLR {row.llr:.1f}", x=0.98, y=0.98,
                           xref=f"x{2 + ci} domain", yref=f"y{2 + ci} domain", showarrow=False,
                           xanchor="right", yanchor="top", font=dict(size=10, color=INK))
        fig.update_yaxes(title_text="fraction of cells" if ci == 1 else None, row=2, col=ci)
        res = binned(d, tm, bins)
        top = res["observed"][0]
        y = np.array(res["observed"]) / top
        fig.add_trace(go.Scatter(x=blab, y=y, mode="markers", marker=dict(symbol="diamond", size=10, color=INK),
                                 error_y=dict(type="data", array=np.array(res["se"]) / top, color=INK, thickness=1.2, width=3),
                                 name="observed", showlegend=ci == 1, legendgroup="obs"), row=3, col=ci)
        for nm, colr, dash in [("gamma 1", "#555555", "dot"), ("gamma 4", CC["buffered"], "solid")]:
            fig.add_trace(go.Scatter(x=blab, y=np.array(res[nm]) / top, mode="lines", line=dict(color=colr, width=2, dash=dash),
                                     name=f"prediction, {nm}", showlegend=ci == 1, legendgroup=nm), row=3, col=ci)
        vals += [dict(panel="3c response", gene=gname, bin=bl, observed=o_, se=se_, g1=a1, g4=a4) for bl, o_, se_, a1, a4 in
                 zip(blab, res["observed"], res["se"], res["gamma 1"], res["gamma 4"])]
        fig.update_xaxes(title_text="target UMI bin" if tm < 5 else "target quintile (low to high)", row=3, col=ci)
        fig.update_yaxes(title_text="response, relative to lowest bin" if ci == 1 else None, row=3, col=ci)
    fig.update_layout(barmode="stack", legend=dict(orientation="h", x=0, y=-0.12))
    style(fig, 1100)
    save(fig, "fig3")
    pd.DataFrame(vals).to_csv(f"{D}/fig3_values.tsv", sep="\t", index=False, float_format="%.4g")


# ---------------------------------------------------------------- figure 4
LC4 = {"K562": "#1F9BB5", "Jurkat": "#B07A12", "RPE1": "#E0457B", "HepG2": "#7B4FB0"}
TP = {"K562": "mutant", "Jurkat": "mutant", "RPE1": "wild type", "HepG2": "wild type"}


def fig4():
    R = {"K562": pd.read_csv(f"{D}/k562_regime.tsv", sep="\t", index_col=0),
         "Jurkat": pd.read_csv("out/jurkat/regime_native.tsv", sep="\t", index_col=0),
         "RPE1": pd.read_csv(f"{D}/rpe1_regime.tsv", sep="\t", index_col=0),
         "HepG2": pd.read_csv("out/hepg2/regime_native.tsv", sep="\t", index_col=0)}
    C = pd.read_csv("out/final/lines4_complex.tsv", sep="\t").set_index("family")
    fig = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5], horizontal_spacing=0.22,
                        subplot_titles=["a  Safe regime, class by method", "b  EM buffered share relative to the line"])
    vals = []
    meth = {"Song PS": "song", "Mixscale": "mix", "Joint EM": "em"}
    ylab = [f"{nm} ({(R[nm].p_safe >= 0.5).sum()})  {m}" for nm in R for m in meth]
    base = np.zeros(len(ylab))
    for c in ["buffered", "sensitive", "not classified"]:
        x = []
        for nm, d in R.items():
            s = d[d.p_safe >= 0.5]
            x += [(cls(s[k]) == c).mean() for k in meth.values()]
        fig.add_trace(go.Bar(y=ylab, x=x, base=base.copy(), orientation="h", name=c, marker_color=CC[c], marker_line=dict(color="white", width=2),
                             text=[f"{v:.0%}" if v >= 0.04 else "" for v in x], textposition="inside", insidetextanchor="middle", textangle=0,
                             textfont=dict(size=10, color=INK), hovertemplate=f"%{{y}}, {c}: %{{x:.3f}}<extra></extra>"), row=1, col=1)
        base += np.array(x)
        vals += [dict(panel="4a", row=yl, cls=c, value=v) for yl, v in zip(ylab, x)]
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=11), row=1, col=1)
    fig.update_xaxes(title_text="fraction of safe-regime groups", range=[0, 1], row=1, col=1)
    fam = C[C[[f"{l}_n" for l in R]].min(axis=1) >= 2].copy()
    fam["mean_rel"] = fam[[f"{l}_rel" for l in R]].mean(axis=1)
    fam = fam.sort_values("mean_rel")
    for nm in R:
        fig.add_trace(go.Scatter(x=fam[f"{nm}_rel"], y=fam.index, mode="markers",
                                 marker=dict(symbol="diamond" if TP[nm] == "mutant" else "circle", size=10, color=LC4[nm], line=dict(color="white", width=1)),
                                 name=f"{nm} (TP53 {TP[nm]})", customdata=np.c_[fam[f"{nm}_n"], fam[f"{nm}_buf"]],
                                 hovertemplate=f"{nm} %{{y}}: %{{x:.2f}} (share %{{customdata[1]:.2f}}, n %{{customdata[0]:.0f}})<extra></extra>"), row=1, col=2)
        vals += [dict(panel="4b", line=nm, family=f, rel=r[f"{nm}_rel"], share=r[f"{nm}_buf"], n=r[f"{nm}_n"]) for f, r in fam.iterrows()]
    fig.add_vline(x=1, line=dict(color="#999999", width=1, dash="dot"), row=1, col=2)
    fig.update_xaxes(title_text="complex buffered share / line buffered share", row=1, col=2)
    fig.update_layout(barmode="overlay", legend=dict(orientation="h", x=0, y=-0.14, traceorder="normal"))
    style(fig, 620)
    fig.update_layout(margin=dict(b=110))
    save(fig, "fig4")
    pd.DataFrame(vals).to_csv(f"{D}/fig4_values.tsv", sep="\t", index=False, float_format="%.4g")


if __name__ == "__main__":
    os.makedirs(D, exist_ok=True)
    for f in (fig2, fig3, fig4, fig1):
        f()
        print("done", f.__name__)


# ---------------------------------------------------------------- figure 5
def fig5():
    gu = pd.read_csv("out/jost/guides.tsv", sep="\t")
    gg = pd.read_csv("out/jost/gene_gamma.tsv", sep="\t").set_index("gene")
    so = pd.read_csv("out/jost/jost_song.genes.tsv", sep="\t").set_index("gene")["class"]
    mi = pd.read_csv("out/jost/jost_mix.tsv", sep="\t").set_index("gene")["class"]
    rep = pd.read_csv("out/gwps/replication.tsv", sep="\t", index_col=0)
    show = ["RPL9", "POLR2H", "HSPA9", "GATA1"]
    fig = make_subplots(rows=2, cols=4, vertical_spacing=0.2, horizontal_spacing=0.1,
                        specs=[[{}, {}, {}, {}], [{"colspan": 2}, None, {"colspan": 2}, None]],
                        subplot_titles=[f"a  {g}" if i == 0 else g for i, g in enumerate(show)] +
                                       ["b  Class of guide-level buffered genes (n 15)", "c  Same class in K562 genome-wide screen"])
    vals = []
    kk = np.linspace(0, 1, 101)
    for ci, g in enumerate(show, start=1):
        x = gu[gu.gene == g]
        fig.add_trace(go.Scatter(x=x.k, y=x.r, mode="markers", marker=dict(symbol="diamond", size=10, color=INK),
                                 error_y=dict(type="data", array=1.96 * x.se, color=INK, thickness=1.2, width=3), showlegend=ci == 1,
                                 name="guide mean", hovertemplate="%{text}: k %{x:.2f}, r %{y:.2f}<extra></extra>", text=x.guide), row=1, col=ci)
        gh = gg.loc[g, "gamma_guide"]
        w = 1 / np.maximum(x.se.values, 1e-3) ** 2
        for gm, nm, col, dash in [(1.0, "gamma 1", "#555555", "dot"), (gh, "fitted gamma", CC["buffered"], "solid")]:
            xg = x.k.clip(0, 1).values ** gm
            A = (w * xg * x.r.values).sum() / (w * xg * xg).sum()
            kx = kk[kk <= x.k.max() + 0.05]
            fig.add_trace(go.Scatter(x=kx, y=A * kx ** gm, mode="lines", line=dict(color=col, width=2, dash=dash), name=nm,
                                     showlegend=ci == 1, legendgroup=nm), row=1, col=ci)
        fig.add_annotation(text=f"gamma {gh:.2f} ({gg.loc[g, 'lo']:.2f} to {gg.loc[g, 'hi']:.2f})", x=0.02, y=0.98,
                           xref=("x domain" if ci == 1 else f"x{ci} domain"), yref=("y domain" if ci == 1 else f"y{ci} domain"), showarrow=False, xanchor="left", yanchor="top", font=dict(size=10, color=INK))
        fig.update_xaxes(title_text="guide knockdown", range=[0, 1], row=1, col=ci)
        fig.update_yaxes(title_text="response" if ci == 1 else None, row=1, col=ci)
        vals += [dict(panel="5a", gene=g, guide=r.guide, k=r.k, r=r.r, se=r.se) for r in x.itertuples()]
    buf = gg[gg.class_guide == "buffered"].index
    meth = {"Song PS": so, "Mixscale": mi, "Joint EM": gg.em_class}
    base = np.zeros(3)
    for c in ["buffered", "sensitive", "not classified"]:
        y = [(cls(meth[m].reindex(buf).fillna("NA")) == c).sum() for m in meth]
        fig.add_trace(go.Bar(y=list(meth), x=y, base=base.copy(), offsetgroup="stack", orientation="h", name=c, marker_color=CC[c], marker_line=dict(color="white", width=2),
                             text=[str(v) if v else "" for v in y], textposition="inside", insidetextanchor="middle", textfont=dict(color=INK, size=11),
                             showlegend=False), row=2, col=1)
        base += np.array(y)
        vals += [dict(panel="5b", method=m, cls=c, n=v) for m, v in zip(meth, y)]
    fig.update_yaxes(autorange="reversed", row=2, col=1)
    fig.update_xaxes(title_text="genes", range=[0, 15], row=2, col=1)
    lab = {"song": "Song PS", "mix": "Mixscale", "em": "Joint EM"}
    for j, (col, nm) in enumerate([("buf_rep", "buffered"), ("sen_rep", "sensitive")]):
        y = [rep.loc[k, col] for k in lab]
        n = [int(rep.loc[k, "n_buf" if nm == "buffered" else "n_sen"]) for k in lab]
        fig.add_trace(go.Bar(x=list(lab.values()), y=y, marker_color=[MC[m] for m in lab.values()], offsetgroup=f"r{j}",
                             marker_pattern_shape="" if j == 0 else "/", showlegend=False,
                             text=[f"{v:.2f} (n {k})" if np.isfinite(v) else f"n {k}" for v, k in zip(y, n)], textposition="outside",
                             textfont=dict(size=10, color=INK)), row=2, col=3)
        vals += [dict(panel="5c", method=m, call=nm, replicated=v, n=k) for m, v, k in zip(lab.values(), y, n)]
    for nm, pat in [("buffered call replicated", ""), ("sensitive call replicated", "/")]:
        fig.add_trace(go.Bar(x=[None], y=[None], name=nm, marker_color="#888888", marker_pattern_shape=pat), row=2, col=3)
    fig.update_yaxes(title_text="fraction of K562 essential calls", range=[0, 0.85], row=2, col=3)
    fig.update_layout(barmode="group", legend=dict(orientation="h", x=0, y=-0.12))
    style(fig, 800)
    save(fig, "fig5")
    pd.DataFrame(vals).to_csv(f"{D}/fig5_values.tsv", sep="\t", index=False, float_format="%.4g")
