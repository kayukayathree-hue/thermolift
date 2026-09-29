"""Build the training dataset: 40 wells x 3-7 CSS cycles x realistic operator recipes + sensor noise.
Outputs data/cycles.csv (one row per cycle) and data/daily.csv (one row per well-cycle-day, 5-day step)."""
import numpy as np, pandas as pd, os
from physics import simulate, cycle_kpis, FIXED
rng = np.random.default_rng(42); OUT = os.path.join(os.path.dirname(__file__), "..", "data")

def make(n_wells=120):
    cyc_rows, day_rows = [], []
    for wi in range(n_wells):
        w = dict(FIXED, id=f"BGW-{wi+1:02d}", depth=rng.uniform(950, 1150), h=rng.uniform(12, 28), q0=rng.uniform(3, 8),
                 wc0=rng.uniform(.55, .75), wcEnd=rng.uniform(.15, .35), tauBase=rng.uniform(80, 150),
                 Kfall=rng.uniform(.04, .09), re=rng.uniform(120, 180), aD=rng.uniform(.7, 1.2))
        for c in range(1, int(rng.integers(4, 9))):
            w["cyc"] = c
            r = dict(V=rng.uniform(4000, 15000), Pinj=rng.uniform(3, 12), rate=rng.uniform(250, 650),
                     x=rng.uniform(.55, .82), soak=int(rng.integers(3, 16)), N0=rng.uniform(2, 8), S=float(rng.choice([2.5, 3.0, 3.5, 4.0])))
            res = simulate(w, r, "base"); cut = int(rng.integers(120, 300)); k = cycle_kpis(res, w, r, cut)
            rf = res["rfi"][:cut]; fail_p = 1 - np.exp(-0.0022 * (rf > 0.85).sum() - 0.0015 * (res["fill"][:cut] < .6).sum() - 0.004 * (rf > 1).sum())
            failed = int(rng.random() < fail_p)  # rod failure / pump unsetting in the cycle
            cyc_rows.append(dict(well=w["id"], cycle=c, depth=w["depth"], pay=w["h"], q0=w["q0"], re=w["re"],
                                 **{f"steam_{a}": r[a] for a in ["V", "Pinj", "rate", "x"]},
                                 soak=r["soak"], SPM=r["N0"], stroke=r["S"], cut=cut, prev_cycles=c - 1,
                                 oil_bbl=k["oil_bbl"] * rng.lognormal(0, .06), SOR=k["SOR"], kWh_bbl=k["kWh_bbl"],
                                 float_days=k["float_days"], max_rfi=float(rf.max()), rod_failure=failed))
            for t in range(0, 365, 5):
                s = lambda a: res[a][t]
                day_rows.append(dict(well=w["id"], cycle=c, day=t + 1, depth=w["depth"], pay=w["h"], steam_V=r["V"], steam_Pinj=r["Pinj"], steam_rate=r["rate"], steam_x=r["x"], q0=w["q0"], re=w["re"], soak=r["soak"], stroke=r["S"],
                                     prev_cycles=c - 1, SPM=res["SPM"][t] * rng.normal(1, .01), wc=s("wc") + rng.normal(0, .01),
                                     T_nearwell=s("T") + rng.normal(0, .8), visc_cP=s("visc_cP") * rng.lognormal(0, .05),
                                     gross_bpd=s("gross") * rng.lognormal(0, .05), oil_bpd=s("oil") * rng.lognormal(0, .06),
                                     kW=s("kW") * rng.normal(1, .02), fill=float(np.clip(s("fill") + rng.normal(0, .02), 0, 1)), rfi=s("rfi") * rng.normal(1, .03)))
    return pd.DataFrame(cyc_rows), pd.DataFrame(day_rows)

if __name__ == "__main__":
    cy, dy = make(); os.makedirs(OUT, exist_ok=True)
    cy.to_csv(os.path.join(OUT, "cycles.csv"), index=False); dy.to_csv(os.path.join(OUT, "daily.csv"), index=False)
    print("cycles", cy.shape, "daily", dy.shape, "| failure rate %.2f" % cy.rod_failure.mean())
