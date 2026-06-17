"""Score a single patient's diabetes risk (educational; not for clinical use).
Example:
    python predict.py --glucose 165 --bmi 33.6 --age 50 --pregnancies 6
"""
import argparse, joblib, pandas as pd
from train import FEATURES

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pregnancies", type=int, default=2)
    ap.add_argument("--glucose", type=float, default=120)
    ap.add_argument("--bloodpressure", type=float, default=70)
    ap.add_argument("--skinthickness", type=float, default=20)
    ap.add_argument("--insulin", type=float, default=80)
    ap.add_argument("--bmi", type=float, default=32)
    ap.add_argument("--pedigree", type=float, default=0.47)
    ap.add_argument("--age", type=int, default=33)
    a = ap.parse_args()
    model = joblib.load("reports/model.joblib")
    row = pd.DataFrame([[a.pregnancies, a.glucose, a.bloodpressure, a.skinthickness,
                         a.insulin, a.bmi, a.pedigree, a.age]], columns=FEATURES)
    print(f"Predicted diabetes risk: {float(model.predict_proba(row)[0,1]):.1%}")
    print("(Educational model on synthetic data — not a medical device.)")

if __name__ == "__main__":
    main()
