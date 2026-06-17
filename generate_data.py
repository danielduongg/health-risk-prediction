"""Generate a Pima-style diabetes dataset (reproducible, no download needed).

Marginal distributions and the outcome's dependence on glucose / BMI / age /
pedigree mirror the well-known Pima Indians Diabetes dataset, so models behave
realistically. We also inject the dataset's famous quirk — physiologically
impossible zeros in Glucose/BloodPressure/BMI/Insulin/SkinThickness — so the
preprocessing step has something real to fix. Swap in the real CSV anytime
(see README); columns match exactly.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

def generate(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = np.clip(rng.gamma(6, 3.0, n) + 21, 21, 81)
    pregnancies = np.clip(rng.poisson(3.0, n), 0, 17)
    glucose = np.clip(rng.normal(120, 32, n), 44, 199)
    bp = np.clip(rng.normal(69, 19, n), 24, 122)
    skin = np.clip(rng.normal(21, 16, n), 0, 99)
    insulin = np.clip(rng.gamma(1.6, 90, n), 0, 846)
    bmi = np.clip(rng.normal(32, 7.0, n), 18, 67)
    pedigree = np.clip(rng.gamma(2.0, 0.25, n), 0.08, 2.5)

    def z(x): return (x - x.mean()) / x.std()
    logit = (-0.9 + 1.15 * z(glucose) + 0.55 * z(bmi) + 0.45 * z(age)
             + 0.30 * z(pedigree) + 0.20 * z(pregnancies) + rng.normal(0, 0.7, n))
    outcome = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    df = pd.DataFrame(dict(
        Pregnancies=pregnancies,
        Glucose=glucose.round(0).astype(int),
        BloodPressure=bp.round(0).astype(int),
        SkinThickness=skin.round(0).astype(int),
        Insulin=insulin.round(0).astype(int),
        BMI=bmi.round(1),
        DiabetesPedigreeFunction=pedigree.round(3),
        Age=age.round(0).astype(int),
        Outcome=outcome,
    ))
    # inject the dataset's signature "impossible zeros" as missing-encoded values
    for col, frac in [("Glucose", .01), ("BloodPressure", .05),
                      ("SkinThickness", .30), ("Insulin", .49), ("BMI", .01)]:
        idx = rng.choice(n, int(frac * n), replace=False)
        df.loc[idx, col] = 0
    return df

if __name__ == "__main__":
    df = generate()
    df.to_csv("data/diabetes.csv", index=False)
    print(f"Wrote {len(df):,} rows | positive rate {df.Outcome.mean():.3f}")
    print((df[["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]] == 0).sum())
