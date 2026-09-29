"""ML-driven optimizers: CSS recipe search + SRP set-point controller."""
import numpy as np, pandas as pd, joblib, json, os
from utils import LogHGR  # noqa (needed to unpickle)
M = os.path.join(os.path.dirname(__file__), "..", "models")
META = json.load(open(f"{M}/metrics.json")); L = lambda n: joblib.load(f"{M}/{n}.joblib")
cycle_oil, T_model, oil_model, rfi_model = L("cycle_oil"), L("T_nearwell"), L("oil_bpd"), L("rfi")
fail_clf, float_clf = L("rod_failure_clf"), L("floating_clf")
ECON = dict(price=70, steam=7, elec=0.10, opex=40)

def forecast(well, recipe, days=np.arange(1, 366)):
    X = pd.DataFrame([dict(well, **recipe, day=d, prev_cycles=well["prev_cycles"]) for d in days])[META["DAY_F"]]
    return dict(day=days, T=T_model.predict(X), oil=np.maximum(oil_model.predict(X), 0))

def _profit_rate(well, r, econ):
    f = forecast(well, r, np.arange(20, 366, 5)); d = f["day"]
    kwh_day = 8.0 * 24 * 0.5                      # nominal lifting load (kWh/d) for economics
    cum = np.cumsum(f["oil"]) * 5; steam = r["steam_V"] * econ["steam"]; down = r["steam_V"] / r["steam_rate"] + r["soak"]
    net = cum * econ["price"] - d * (econ["opex"] + kwh_day * econ["elec"]) - steam
    pr = net / (down + d); k = int(np.argmax(pr)); return pr[k], int(d[k]), float(cum[k])

def css_optimize(well, current, econ=ECON):
    """Grid search over steam volume / soak / injection pressure; cut-off is the profit-rate-optimal day."""
    best, rows = None, []
    for V in np.arange(4000, 15001, 1000):
        for soak in (3, 5, 7, 10, 14):
            for P in (4, 6, 8, 10):
                r = dict(current, steam_V=V, soak=soak, steam_Pinj=P); pr, cut, oil = _profit_rate(well, r, econ)
                rows.append(dict(V=V, soak=soak, Pinj=P, cut=cut, oil=oil, SOR=V / max(oil, 1), profit_rate=pr))
    df = pd.DataFrame(rows); return df, df.loc[df.profit_rate.idxmax()]

def srp_optimize(visc, wc, gross_bpd, stroke, rfi_limit=0.6):
    """Highest-throughput SPM whose predicted RFI stays below the limit; if the demanded SPM would exceed it, run on a duty cycle."""
    cap1 = np.pi / 4 * 0.044 ** 2 * stroke * .85 * 1440 * 6.2898           # bbl/d per SPM
    need = gross_bpd / (cap1 * .85); grid = np.round(np.arange(1, 10.01, 0.25), 2)
    X = pd.DataFrame(dict(visc_cP=visc, SPM=grid, stroke=stroke, gross_bpd=gross_bpd, wc=wc))[META["RFI_F"]]
    rfi = rfi_model.predict(X); ok = grid[rfi <= rfi_limit]
    spm = float(min(max(ok.max() if len(ok) else 1.0, 1.0), max(need, 1.0)) if len(ok) else 1.0)
    duty = float(min(1.0, need / spm)); r_at = float(rfi_model.predict(X.iloc[[int(np.argmin(abs(grid - spm)))]])[0])
    return dict(SPM=spm, duty=duty, rfi=r_at, need=float(need))
