import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import streamlit as st, pandas as pd, numpy as np, plotly.graph_objects as go
import optimizer as O; from physics import simulate, cycle_kpis, FIXED, K
st.set_page_config("ThermoLift Digital Twin", layout="wide")
st.title("ThermoLift — CSS + SRP Well-to-Surface Digital Twin")
st.caption("Baghewala heavy-oil prototype · ML models trained on a physics-generated synthetic history (see README for real-data retraining)")
sb = st.sidebar; sb.header("Well & current practice")
well = dict(depth=sb.slider("Depth (m)", 950, 1150, 1050), pay=sb.slider("Net pay (m)", 12, 28, 20), q0=sb.slider("Cold rate q0 (bbl/d)", 3.0, 8.0, 5.5),
            re=150, prev_cycles=sb.slider("Previous CSS cycles", 0, 6, 2))
cur = dict(steam_V=sb.slider("Steam volume (bbl)", 4000, 15000, 9000, 500), steam_Pinj=sb.slider("Injection pressure (MPa)", 3.0, 12.0, 5.0, .5),
           steam_rate=sb.slider("Injection rate (bbl/d)", 250, 650, 400, 25), steam_x=sb.slider("Steam quality", .55, .82, .70, .01),
           soak=sb.slider("Soak (d)", 3, 15, 7), SPM=sb.slider("Pump speed today (SPM)", 2.0, 8.0, 5.0, .5), stroke=sb.select_slider("Stroke (m)", [2.5, 3.0, 3.5, 4.0], 3.0))
econ = dict(price=sb.number_input("Oil $/bbl", 30, 150, 70), steam=sb.number_input("Steam $/bbl", 2.0, 20.0, 7.0), elec=0.10, opex=40)
t1, t2, t3, t4 = st.tabs(["Digital Twin", "CSS Advisor", "SRP Advisor", "Model performance"])

with t1:
    st.subheader("Reservoir heating/cooling and production forecast (ML) vs physics model")
    f = O.forecast(well, cur)
    w = dict(FIXED, depth=well["depth"], h=well["pay"], q0=well["q0"], cyc=well["prev_cycles"] + 1, re=150, wc0=.65, wcEnd=.25, aD=1.0)
    ph = simulate(w, dict(V=cur["steam_V"], Pinj=cur["steam_Pinj"], rate=cur["steam_rate"], x=cur["steam_x"], soak=cur["soak"], N0=cur["SPM"], S=cur["stroke"]), "base")
    c1, c2 = st.columns(2)
    for col, key, ref, title in [(c1, "T", "T", "Near-well temperature (°C)"), (c2, "oil", "oil", "Oil rate (bbl/d)")]:
        fg = go.Figure([go.Scatter(x=f["day"], y=f[key], name="ML forecast", line=dict(width=3)), go.Scatter(x=ph["day"], y=ph[ref], name="Physics model", line=dict(dash="dot"))])
        fg.update_layout(title=title, height=330, margin=dict(t=40, b=10), xaxis_title="Production day"); col.plotly_chart(fg, width='stretch')
    k = cycle_kpis(ph, w, dict(V=cur["steam_V"], rate=cur["steam_rate"], soak=cur["soak"]))
    a, b, c, d = st.columns(4); a.metric("Cycle oil (ML)", f"{O.cycle_oil.predict(pd.DataFrame([dict(well, **cur, cut=220)])[O.META['CYC_F']])[0]:,.0f} bbl")
    b.metric("SOR (physics)", f"{k['SOR']:.2f}"); c.metric("kWh / bbl", f"{k['kWh_bbl']:.1f}"); d.metric("Days rod-floating", k["float_days"])
    fig = go.Figure([go.Scatter(x=ph["day"], y=ph["rfi"], name="Rod-floating index")]); fig.add_hline(y=1, line_color="red", annotation_text="floating"); fig.add_hline(y=.6, line_color="green", annotation_text="safe target")
    fig.update_layout(title="Rod floating index at current SPM", height=300, margin=dict(t=40, b=10)); st.plotly_chart(fig, width='stretch')

with t2:
    if True:
        df, b = O.css_optimize(well, cur, econ); base_pr, base_cut, base_oil = O._profit_rate(well, cur, econ)
        st.success(f"Recommended: **{b.V:,.0f} bbl** steam · **{b.Pinj:.0f} MPa** · soak **{b.soak:.0f} d** · cut-off **day {b.cut:.0f}**")
        a, c, d = st.columns(3)
        a.metric("Profit rate ($/cycle-day)", f"{b.profit_rate:,.0f}", f"{b.profit_rate - base_pr:+,.0f}")
        c.metric("SOR", f"{b.SOR:.2f}", f"{b.SOR - cur['steam_V'] / base_oil:+.2f}", delta_color="inverse")
        d.metric("Cycle oil (bbl)", f"{b.oil:,.0f}", f"{b.oil - base_oil:+,.0f}")
        pv = df.groupby(["V", "soak"]).profit_rate.max().unstack()
        st.plotly_chart(go.Figure(go.Heatmap(z=pv.values, x=pv.columns, y=pv.index, colorscale="Tealgrn", colorbar=dict(title="$/d"))).update_layout(
            title="Profit-rate map: steam volume × soak (best pressure)", xaxis_title="Soak (d)", yaxis_title="Steam (bbl)", height=380), width='stretch')
        st.dataframe(df.sort_values("profit_rate", ascending=False).head(10).round(2), hide_index=True)

with t3:
    st.subheader("Live SRP set-point control")
    day = st.slider("Production day", 1, 365, 60); ph2 = {k_: v[day - 1] for k_, v in ph.items() if np.ndim(v) == 1}
    visc, wc, gross = ph2["visc_cP"], ph2["wc"], ph2["gross"]; ctl = O.srp_optimize(visc, wc, gross, cur["stroke"])
    cur_rfi = float(O.rfi_model.predict(pd.DataFrame(dict(visc_cP=[visc], SPM=[cur["SPM"]], stroke=[cur["stroke"]], gross_bpd=[gross], wc=[wc]))[O.META["RFI_F"]])[0])
    p_float = float(O.float_clf.predict_proba(pd.DataFrame(dict(visc_cP=[visc], SPM=[cur["SPM"]], stroke=[cur["stroke"]], gross_bpd=[gross], wc=[wc]))[O.META["RFI_F"]])[0, 1])
    a, b, c, d = st.columns(4); a.metric("Viscosity at rods", f"{visc:,.0f} cP"); b.metric("Current SPM → recommended", f"{cur['SPM']:.1f} → {ctl['SPM']:.2f}")
    c.metric("Predicted RFI", f"{cur_rfi:.2f} → {ctl['rfi']:.2f}", delta_color="inverse"); d.metric("P(rod floating) now", f"{p_float:.0%}")
    (st.error if cur_rfi > 1 else st.warning if cur_rfi > .6 else st.success)(f"Rod-floating index {cur_rfi:.2f}: " + ("FLOATING — impact loading likely; reduce SPM" if cur_rfi > 1 else "watch" if cur_rfi > .6 else "safe"))
    if ctl["duty"] < 1: st.info(f"Well cannot fill the pump at a safe speed: run **{ctl['SPM']:.2f} SPM on a {ctl['duty']:.0%} VFD duty cycle** (avoids fluid pound).")
    spm = np.arange(1, 10.01, .25); Xs = pd.DataFrame(dict(visc_cP=visc, SPM=spm, stroke=cur["stroke"], gross_bpd=gross, wc=wc))[O.META["RFI_F"]]
    fg = go.Figure(go.Scatter(x=spm, y=O.rfi_model.predict(Xs), name="RFI")); fg.add_hline(y=.6, line_color="green"); fg.add_hline(y=1, line_color="red")
    fg.update_layout(title="Rod floating index vs pump speed (today's fluid)", xaxis_title="SPM", height=320); st.plotly_chart(fg, width='stretch')

with t4:
    m = O.META; st.write(f"Training data: **{m['n_wells']} wells · {m['n_cycles']} CSS cycles · {m['n_daily']:,} well-days**. All scores are 5-fold cross-validated **by well** (unseen wells).")
    st.dataframe(pd.DataFrame(m["metrics"]).T, width='stretch')
    fi = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "cycles.csv")); st.caption("Sample of training data (cycles.csv)"); st.dataframe(fi.head(30), hide_index=True)
