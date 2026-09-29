"""Physics core of the Digital Twin (Python port of the ThermoLift design model).
Reservoir heat decay -> viscosity -> inflow -> rod-fall velocity -> SRP loading."""
import numpy as np
K = dict(Tr=47, cp=4.4, M=2300, cwe=158.987, rhoF=980, rhoS=7850, g=9.81, H=365)
FIXED = dict(rw=0.1, re=150, wc0=0.65, wcEnd=0.25, aD=1.0, pumpD=0.044, rodD=0.022,
             tubeD=0.062, rodFrac=0.5, dmgCost=6000, tauBase=110, Kfall=0.06,
             price=70, steamCost=7, elec=0.10, opex=40)

def tsat(P): return 1810.94 / (8.14019 - np.log10(P * 7500.62)) - 244.485
def hfg(T): return 2257 * max(374.15 - T, 1) ** 0.38 / 274.15 ** 0.38
def andrade(m1, T1, m2, T2):
    a, b = 1 / (T1 + 273.15), 1 / (T2 + 273.15)
    B = np.log(m1 / m2) / (a - b); return np.log(m1) - B * a, B
VM = andrade(800, 47, 40, 100)          # 800 cP @47C (heavy 17-19 API), 40 cP @100C
def mu_at(T, vm=VM): return np.exp(vm[0] + vm[1] / (np.asarray(T) + 273.15))

def simulate(w, r, mode="ctrl", vm=VM):
    """w: well dict, r: recipe dict(V,Pinj,rate,x,soak,N0,S). Returns dict of daily arrays (365d)."""
    H, Tr = K["H"], K["Tr"]
    Ti = tsat(r["Pinj"]); hK = r["x"] * hfg(Ti) + K["cp"] * (Ti - Tr)
    loss = min(0.5, (0.06 + 0.00012 * w["depth"]) * (400 / max(r["rate"], 50)) ** 0.35)
    Q = r["V"] * K["cwe"] * hK * (1 - loss); dT = 0.55 * (Ti - Tr)
    rh = np.clip(np.sqrt(Q * 0.8 / (K["M"] * dT) / (np.pi * w["h"])), w["rw"] * 3, w["re"] * 0.9)
    tau = w["tauBase"] * np.sqrt(w["h"] / 20) * (rh / 10) ** 0.5
    drv = w["aD"] * (rh ** 2 / 100) ** 0.8 * np.exp(-r["soak"] / 45)
    T0 = Tr + dT * np.exp(-r["soak"] / (1.5 * tau)); soakF = 0.7 + 0.3 * (1 - np.exp(-r["soak"] / 3))
    muc = float(mu_at(Tr, vm)); lnRe, lnRh, lnRR = np.log(w["re"] / w["rw"]), np.log(rh / w["rw"]), np.log(w["re"] / rh)
    dep = 0.93 ** (w["cyc"] - 1)
    Ap = np.pi / 4 * w["pumpD"] ** 2; S = r["S"]; cap1 = Ap * S * 0.85 * 1440 * 6.2898
    Ar = np.pi / 4 * w["rodD"] ** 2; lnDD = np.log(w["tubeD"] / w["rodD"]); L = w["depth"] - 30
    td = np.arange(1, H + 1)
    Tav = Tr + (T0 - Tr) * np.exp(-td / tau); muh = mu_at(Tav, vm)
    ratio = muc * lnRe / (muh * lnRh + muc * lnRR)
    qo = w["q0"] * w.get("calib", 1) * dep * ratio * (1 + drv * np.exp(-td / 40)) * soakF
    wc = w["wcEnd"] + (w["wc0"] - w["wcEnd"]) * np.exp(-td / 45); qg = qo / (1 - wc)
    mur = mu_at(Tr + w["rodFrac"] * (Tav - Tr), vm) * (1 + 1.2 * wc) / 1000
    vf = w["Kfall"] * (K["rhoS"] - K["rhoF"]) * K["g"] * Ar * lnDD / (2 * np.pi * mur)  # rod fall velocity
    N = np.full(H, float(r["N0"])); fd = np.full(H, 0.5); duty = np.ones(H)
    if mode == "ctrl":                                   # rule-based controller
        for t in range(H):
            Nreq = qg[t] / (cap1 * 0.85); n = np.clip(Nreq, 1, 10); d = min(Nreq, 1) if Nreq < 1 else 1; f = 0.5
            if np.pi * S * n / (120 * f) / vf[t] > 0.6:
                f = np.clip(np.pi * S * n / (120 * 0.6 * vf[t]), 0.5, 0.65)
                if np.pi * S * n / (120 * f) / vf[t] > 0.6:
                    n = max(1, 0.6 * vf[t] * 120 * f / (np.pi * S)); d = min(1, Nreq / n)
            N[t], fd[t], duty[t] = n, f, d
    idx = np.pi * S * N / (120 * fd) / vf
    capEff = cap1 * N * duty * (1 - 0.5 * np.clip((idx - 0.85) / 0.35, 0, 1))
    fill = np.minimum(1, qg / capEff); prodG = np.minimum(qg, capEff); prodO = qo * prodG / qg
    pound = np.where(fill < 0.7, (0.7 - fill) / 0.7, 0)
    Phyd = (prodG * 0.158987 / 86400) * K["rhoF"] * K["g"] * (0.85 * L) / 1000
    vavg = 2 * S * N / 60; Pvisc = 2 * np.pi * mur * L * vavg ** 2 / lnDD / 1000; Pfric = 0.22 * N * (S / 3) * (L / 1000)
    kW = (Phyd + (Pvisc + Pfric) * duty) / 0.72 + 0.4
    dam = N * duty * 1440 * (1 + 4 * np.maximum(0, idx - 0.85) + 1.5 * pound) / 1e6
    net = prodO * w["price"] - kW * 24 * w["elec"] - w["opex"] - dam * w["dmgCost"]
    return dict(day=td, T=Tav, visc_cP=mur * 1000, vf=vf, q_reservoir=qo, oil=prodO, gross=prodG, wc=wc,
                SPM=N, duty=duty, fd=fd, rfi=idx, fill=fill, kW=kW, damage=dam, net=net, rh=rh, T0=T0, Ti=Ti)

def cycle_kpis(res, w, r, cut=None):
    """Economics of one cycle; cut=None -> profit-rate-optimal cut-off."""
    H = K["H"]; steam = r["V"] * w["steamCost"]; down = r["V"] / r["rate"] + r["soak"]
    cum = np.cumsum(res["net"]); pr = (cum - steam) / (down + np.arange(1, H + 1))
    tc = int(np.argmax(pr[19:]) + 19) if cut is None else int(np.clip(cut - 1, 0, H - 1))
    oil = float(np.cumsum(res["oil"])[tc])
    return dict(cutoff_day=tc + 1, oil_bbl=oil, SOR=r["V"] / max(oil, 1), net_usd=float(cum[tc] - steam),
                profit_rate=float(pr[tc]), kWh_bbl=float((res["kW"][:tc + 1] * 24).sum() / max(oil, 1)),
                float_days=int((res["rfi"][:tc + 1] > 1).sum()), high_days=int((res["rfi"][:tc + 1] > .85).sum()))
