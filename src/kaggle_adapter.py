"""Map a real / Kaggle CSV (e.g. Volve production data, oil-well production tables) into the training schema.
Edit COLMAP once, run `python kaggle_adapter.py file.csv`, then `python train.py` to retrain on real data."""
import sys, pandas as pd, os
COLMAP = {  # <your column> : <schema column>
 "WELL_NAME": "well", "DATEPRD": "date", "BORE_OIL_VOL": "oil_bpd", "BORE_WAT_VOL": "water_bpd", "BORE_GAS_VOL": "gas",
 "AVG_DOWNHOLE_TEMPERATURE": "T_nearwell", "AVG_WHP_P": "whp", "ON_STREAM_HRS": "hours"}
def convert(path, out="../data/daily.csv"):
    d = pd.read_csv(path).rename(columns=COLMAP); d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["well", "date"]); d["day"] = d.groupby("well").cumcount() + 1
    d["gross_bpd"] = d.get("oil_bpd", 0) + d.get("water_bpd", 0); d["wc"] = d.get("water_bpd", 0) / d.gross_bpd.clip(lower=1e-6)
    d.to_csv(os.path.join(os.path.dirname(__file__), out), index=False); print("wrote", out, d.shape, "- fill CSS/SRP columns listed in README")
if __name__ == "__main__": convert(sys.argv[1])
