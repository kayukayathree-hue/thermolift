"""Train the Digital-Twin ML layer. Wells are split by GROUP (well) so scores reflect unseen wells."""
import pandas as pd, numpy as np, joblib, json, os
from sklearn.ensemble import HistGradientBoostingRegressor as HGR, HistGradientBoostingClassifier as HGC
from sklearn.model_selection import GroupKFold, cross_val_predict
from utils import LogHGR
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score, f1_score
D = os.path.join(os.path.dirname(__file__), "..", "data"); M = os.path.join(os.path.dirname(__file__), "..", "models"); os.makedirs(M, exist_ok=True)
cy, dy = pd.read_csv(f"{D}/cycles.csv"), pd.read_csv(f"{D}/daily.csv")

CYC_F = ["depth", "pay", "q0", "re", "steam_V", "steam_Pinj", "steam_rate", "steam_x", "soak", "SPM", "stroke", "cut", "prev_cycles"]
DAY_F = ["day", "depth", "pay", "q0", "re", "steam_V", "steam_Pinj", "steam_rate", "steam_x", "soak", "stroke", "prev_cycles", "SPM"]                 # forecast: known at planning time
RFI_F = ["visc_cP", "SPM", "stroke", "gross_bpd", "wc"]                                          # live sensors -> rod floating index
metrics = {}
def cv(model, X, y, g, kind):
    gkf = GroupKFold(5)
    if kind == "reg":
        p = cross_val_predict(model, X, y, groups=g, cv=gkf); return dict(R2=round(r2_score(y, p), 3), MAE=round(mean_absolute_error(y, p), 3))
    p = cross_val_predict(model, X, y, groups=g, cv=gkf, method="predict_proba")[:, 1]
    return dict(AUC=round(roc_auc_score(y, p), 3), F1=round(f1_score(y, p > .5), 3), positive_rate=round(float(y.mean()), 3))

# 1) CSS cycle recovery model  (steam recipe + well -> cumulative oil)
m = HGR(max_iter=300, learning_rate=.06, random_state=0); metrics["cycle_oil"] = cv(m, cy[CYC_F], cy.oil_bbl, cy.well, "reg")
joblib.dump(m.fit(cy[CYC_F], cy.oil_bbl), f"{M}/cycle_oil.joblib")
# 2) Daily reservoir heating/cooling + production forecast (near-well T, oil rate)
for tgt in ["T_nearwell", "oil_bpd"]:
    m = HGR(max_iter=300, learning_rate=.06, random_state=0); metrics[tgt] = cv(m, dy[DAY_F], dy[tgt], dy.well, "reg")
    joblib.dump(m.fit(dy[DAY_F], dy[tgt]), f"{M}/{tgt}.joblib")
# 3) Rod Floating Index estimator (soft sensor) + floating classifier
m = LogHGR(max_iter=300, learning_rate=.06, random_state=0); metrics["rfi"] = cv(m, dy[RFI_F], dy.rfi, dy.well, "reg")
joblib.dump(m.fit(dy[RFI_F], dy.rfi), f"{M}/rfi.joblib")
dy["floating"] = (dy.rfi > 1).astype(int); c = HGC(max_iter=200, random_state=0)
metrics["rod_floating_flag"] = cv(c, dy[RFI_F], dy.floating, dy.well, "clf"); joblib.dump(c.fit(dy[RFI_F], dy.floating), f"{M}/floating_clf.joblib")
# 4) Rod-failure / pump-unsetting risk per cycle (predictive maintenance)
FAIL_F = CYC_F + ["max_rfi", "float_days"]
c = HGC(max_iter=150, learning_rate=.05, max_depth=3, random_state=0); metrics["rod_failure"] = cv(c, cy[FAIL_F], cy.rod_failure, cy.well, "clf")
joblib.dump(c.fit(cy[FAIL_F], cy.rod_failure), f"{M}/rod_failure_clf.joblib")
json.dump(dict(metrics=metrics, n_wells=int(cy.well.nunique()), n_cycles=len(cy), n_daily=len(dy), CYC_F=CYC_F, DAY_F=DAY_F, RFI_F=RFI_F, FAIL_F=FAIL_F),
          open(f"{M}/metrics.json", "w"), indent=1)
print(json.dumps(metrics, indent=1))
