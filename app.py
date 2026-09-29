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

def heat_color(T):
    """Same colour ramp as the original design: cool blue -> amber -> red as near-well temperature rises."""
    stops = [(47, (43, 108, 176)), (80, (86, 160, 190)), (110, (232, 190, 70)), (140, (232, 120, 50)), (175, (200, 40, 32))]
    T = min(max(T, 47), 175)
    for i in range(1, len(stops)):
        a, b = stops[i - 1], stops[i]
        if T <= b[0]:
            fr = (T - a[0]) / (b[0] - a[0])
            return "rgb(%d,%d,%d)" % tuple(round(av + (bv - av) * fr) for av, bv in zip(a[1], b[1]))
    return "rgb(200,40,32)"

def well_diagram_svg(T, idx, rh):
    """Cross-section: surface, rod string coloured by rod-floating risk, heated zone sized by radius, pump."""
    rx = min(max(rh * 4.5, 18), 100)
    rc = "#ef4444" if idx > 1 else "#f0653a" if idx > 0.85 else "#eab308" if idx > 0.6 else "#22c55e"
    hz1, hz2 = heat_color(T + 25), heat_color(T)
    return f'''<svg viewBox="0 0 240 330" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;background:#141c26;border-radius:8px">
      <defs><radialGradient id="hz"><stop offset="0" stop-color="{hz1}"/><stop offset="1" stop-color="{hz2}" stop-opacity=".35"/></radialGradient></defs>
      <rect x="0" y="62" width="240" height="268" fill="#141c26"/><rect x="0" y="238" width="240" height="72" fill="{heat_color(47)}" opacity=".22"/>
      <ellipse cx="120" cy="274" rx="{rx:.0f}" ry="34" fill="url(#hz)"/>
      <line x1="0" x2="240" y1="62" y2="62" stroke="#93a6b3" stroke-width="2"/>
      <rect x="20" y="52" width="26" height="10" fill="#93a6b3"/><path d="M32 52L72 30L112 34" stroke="#e3edf3" stroke-width="4" fill="none"/><path d="M32 52L72 30" stroke="#e3edf3" stroke-width="4"/>
      <line x1="108" x2="108" y1="34" y2="62" stroke="#e3edf3" stroke-width="1.5"/>
      <line x1="112" x2="112" y1="62" y2="312" stroke="#4b5b68" stroke-width="2"/><line x1="128" x2="128" y1="62" y2="312" stroke="#4b5b68" stroke-width="2"/>
      <line x1="120" x2="120" y1="34" y2="258" stroke="{rc}" stroke-width="3"/>
      <rect x="114" y="258" width="12" height="30" rx="2" fill="{rc}" stroke="#e3edf3"/>
      <text x="8" y="84" fill="#e3edf3" font-size="11">Surface</text><text x="8" y="256" fill="#e3edf3" font-size="11">Heated zone</text>
      <text x="150" y="150" fill="#e3edf3" font-size="11">Rod string</text><text x="150" y="163" font-size="11" font-weight="600" fill="{rc}">index {idx:.2f}</text>
      <text x="8" y="298" fill="#e3edf3" font-size="11" font-weight="600">{T:.0f} \u00b0C avg</text><text x="8" y="311" fill="#e3edf3" font-size="11">radius \u2248 {rh:.0f} m</text>
      <text x="150" y="276" fill="#e3edf3" font-size="11">Pump</text></svg>'''

def rfi_gauge_html(idx):
    pct = min(max(idx, 0), 1.5) / 1.5 * 100
    label = "floating" if idx > 1 else "high" if idx > 0.85 else "watch" if idx > 0.6 else "safe"
    color = "#ef4444" if idx > 1 else "#f0653a" if idx > 0.85 else "#eab308" if idx > 0.6 else "#22c55e"
    return f'''<div style="margin-top:6px"><div style="display:flex;justify-content:space-between;font-size:12.5px;color:#93a6b3">
      <span>Rod floating index, current practice</span><span style="color:{color};font-weight:600">{idx:.2f} \u00b7 {label}</span></div>
      <div style="position:relative;height:16px;border-radius:99px;margin:8px 0 4px;background:linear-gradient(90deg,#22c55e 0 40%,#eab308 40% 57%,#ef4444 57% 100%)">
      <div style="position:absolute;top:-6px;left:{pct:.1f}%;width:4px;height:28px;background:#e3edf3;border-radius:2px;transform:translateX(-2px)"></div></div>
      <div style="display:flex;justify-content:space-between;font-size:11px;color:#93a6b3"><span>0</span><span>0.6 watch</span><span>0.85 high</span><span>1.0 floating</span><span>1.5</span></div></div>'''

def alerts_html(ph, ph_ctrl, well_, r0, d):
    """Same rules as the original design's alertsList(): rod floating, fluid pound, injection pressure, and a benefit summary."""
    out = []
    idx_to_d = ph["rfi"][:d + 1]
    d1 = next((i + 1 for i, v in enumerate(ph["rfi"]) if v > 0.85), None)
    d2 = next((i + 1 for i, v in enumerate(ph["rfi"]) if v > 1.0), None)
    if ph["rfi"][d] > 1:
        out.append(("high", f"Rod floating today (index {ph['rfi'][d]:.2f}). Reduce SPM or slow the downstroke now."))
    elif d2:
        out.append(("high", f"Rod floating expected from day {d2} at the current pump setting. Plan to cut SPM before then."))
    elif d1:
        out.append(("med", f"Floating index passes 0.85 on day {d1}. Tighten the downstroke speed."))
    if (ph["fill"][:d + 1] < 0.7).any():
        pd_ = int(np.argmax(ph["fill"][:d + 1] < 0.7)) + 1
        out.append(("med", f"Fluid pound risk from day {pd_}: fixed {r0['N0']:.1f} SPM outruns inflow, fillage falls below 70%."))
    Pf = 0.012 * well_["depth"]
    if r0["Pinj"] > 0.85 * Pf:
        out.append(("high", f"Injection pressure {r0['Pinj']:.1f} MPa exceeds 85% of the estimated fracture pressure ({Pf:.1f} MPa)."))
    inj_days = r0["V"] / r0["rate"]
    if inj_days > 35:
        out.append(("med", f"Injection takes {inj_days:.0f} days at {r0['rate']:.0f} bbl/d. Consider a higher rate if generator capacity allows."))
    gain = ph_ctrl["oil"][d] - ph["oil"][d]
    if gain > 0.05:
        out.append(("info", f"ThermoLift's SPM control would lift today's rate by {gain:.1f} bbl/d ({gain/ph['oil'][d]*100:.0f}%) at the same steam recipe."))
    if not out:
        out.append(("info", "No active alerts: rod loading and fillage are within safe range today."))
    css = dict(high="#ef4444", med="#eab308", info="#3b82f6")
    return "".join(f'<div style="display:flex;gap:10px;padding:8px 10px;border-radius:8px;background:#141c26;'
                   f'border-left:4px solid {css[lvl]};font-size:13px;color:#e3edf3;margin-bottom:6px">{msg}</div>' for lvl, msg in out)

with t1:
    w = dict(FIXED, depth=well["depth"], h=well["pay"], q0=well["q0"], cyc=well["prev_cycles"] + 1, re=150, wc0=.65, wcEnd=.25, aD=1.0)
    r0 = dict(V=cur["steam_V"], Pinj=cur["steam_Pinj"], rate=cur["steam_rate"], x=cur["steam_x"], soak=cur["soak"], N0=cur["SPM"], S=cur["stroke"])
    ph = simulate(w, r0, "base")                       # current practice (fixed pump speed)
    ph_ctrl = simulate(w, r0, "ctrl")                   # ThermoLift-controlled pump speed, same steam recipe

    st.subheader("Production day")
    st.caption("Drag the slider to see the heated zone cool and the rod-string risk change day by day.")
    day = st.slider("Production day", 1, 365, 60, key="day_twin", label_visibility="collapsed")
    st.markdown(f"**Day {day}**")
    d = day - 1
    vavg_plunger = 2 * cur["stroke"] * ph["SPM"][d] / 60
    cum_oil = float(np.cumsum(ph["oil"])[d]); cum_sor = r0["V"] / max(cum_oil, 1)

    c1, c2 = st.columns([1, 1.6], gap="large")
    with c1:
        st.markdown(well_diagram_svg(ph["T"][d], ph["rfi"][d], float(ph["rh"])), unsafe_allow_html=True)
    with c2:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Near-well temperature", f"{ph['T'][d]:.0f} °C", "current recipe")
        s2.metric("Viscosity at rods", f"{ph['visc_cP'][d]:,.0f} cP", "with emulsion")
        s3.metric("Rod fall speed", f"{ph['vf'][d]:.2f} m/s", f"plunger {vavg_plunger:.2f} m/s")
        s4.metric("Oil rate today", f"{ph['oil'][d]:.1f} bbl/d", f"ThermoLift: {ph_ctrl['oil'][d]:.1f} bbl/d")
        st.markdown(rfi_gauge_html(float(ph["rfi"][d])), unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        s5, s6, s7, s8 = st.columns(4)
        s5.metric("Water cut", f"{ph['wc'][d]*100:.0f} %")
        s6.metric("Gross liquid", f"{ph['gross'][d]:.1f} bbl/d")
        s7.metric("Pump fillage", f"{ph['fill'][d]*100:.0f} %")
        s8.metric("Power draw", f"{ph['kW'][d]:.1f} kW")
        st.caption(f"Cumulative to day {day}: **{cum_oil:,.0f} bbl** oil · running SOR **{cum_sor:.2f}** · pump duty **{ph['duty'][d]*100:.0f}%**")

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        st.markdown("**Alerts**")
        st.markdown(alerts_html(ph, ph_ctrl, well, r0, d), unsafe_allow_html=True)

    st.divider()
    st.subheader("Reservoir heating/cooling and production forecast (ML) vs physics model")
    f = O.forecast(well, cur)
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