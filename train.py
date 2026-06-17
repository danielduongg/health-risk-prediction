"""Train diabetes-risk models with clinically-minded evaluation:
cross-validated AUC, a calibration-method comparison (sigmoid vs isotonic),
permutation importance, and persisted holdout predictions for the fairness audit.
"""
from __future__ import annotations
import json, logging, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             precision_recall_curve, brier_score_loss,
                             classification_report)
import xgboost as xgb

log = logging.getLogger("health.train")
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
REPORTS = "reports"


def load_clean(path="data/diabetes.csv"):
    df = pd.read_csv(path)
    df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)
    return df


def run():
    os.makedirs(REPORTS, exist_ok=True)
    df = load_clean()
    X, y = df[FEATURES], df["Outcome"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    base = Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=2000))])
    xgb_model = Pipeline([("impute", SimpleImputer(strategy="median")),
        ("clf", xgb.XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=42))])

    # cross-validated AUC (model selection done honestly on the training split)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    cv_auc = {"logistic": round(float(cross_val_score(base, Xtr, ytr, cv=cv, scoring="roc_auc").mean()), 4),
              "xgboost": round(float(cross_val_score(xgb_model, Xtr, ytr, cv=cv, scoring="roc_auc").mean()), 4)}

    # calibration-method comparison on the logistic model
    calib = {}
    for method in ("sigmoid", "isotonic"):
        m = CalibratedClassifierCV(base, method=method, cv=5).fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        calib[method] = round(float(brier_score_loss(yte, p)), 4)
    best_method = min(calib, key=calib.get)

    models = {"logistic_regression": CalibratedClassifierCV(base, method=best_method, cv=5),
              "xgboost": xgb_model}
    metrics, preds = {}, {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        preds[name] = p
        rep = classification_report(yte, (p > 0.5).astype(int), output_dict=True, zero_division=0)
        metrics[name] = dict(
            roc_auc=round(float(roc_auc_score(yte, p)), 4),
            pr_auc=round(float(average_precision_score(yte, p)), 4),
            recall_pos=round(float(rep["1"]["recall"]), 3),
            precision_pos=round(float(rep["1"]["precision"]), 3),
            brier=round(float(brier_score_loss(yte, p)), 4))

    perm = permutation_importance(models["xgboost"], Xte, yte, n_repeats=15,
                                  random_state=42, scoring="roc_auc")
    imp = pd.Series(perm.importances_mean, index=FEATURES).sort_values()
    _plots(yte, preds, metrics, imp)

    out = Xte.copy(); out["Outcome"] = yte.values; out["risk_xgb"] = preds["xgboost"]
    out["risk_logreg"] = preds["logistic_regression"]
    out.to_csv(f"{REPORTS}/test_predictions.csv", index=False)
    import joblib; joblib.dump(models["logistic_regression"], f"{REPORTS}/model.joblib")

    summary = dict(cv_auc=cv_auc, calibration_brier=calib,
                   calibration_method_used=best_method, holdout=metrics)
    with open(f"{REPORTS}/metrics.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def _plots(yte, preds, metrics, imp):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(yte, p); ax[0].plot(fpr, tpr, label=f"{name} ({metrics[name]['roc_auc']})")
        pr, rc, _ = precision_recall_curve(yte, p); ax[1].plot(rc, pr, label=f"{name} ({metrics[name]['pr_auc']})")
        fp, mp = calibration_curve(yte, p, n_bins=8); ax[2].plot(mp, fp, "o-", label=name)
    ax[0].plot([0,1],[0,1],"k--",alpha=.4); ax[0].set_title("ROC"); ax[0].legend()
    ax[1].axhline(yte.mean(),ls="--",c="k",alpha=.4); ax[1].set_title("Precision–Recall"); ax[1].legend()
    ax[2].plot([0,1],[0,1],"k--",alpha=.4); ax[2].set_title("Calibration"); ax[2].legend()
    plt.tight_layout(); plt.savefig(f"{REPORTS}/model_curves.png", dpi=120); plt.close()
    plt.figure(figsize=(6,4)); imp.plot.barh(color="#2b6cb0")
    plt.title("Permutation importance (ROC-AUC drop)"); plt.tight_layout()
    plt.savefig(f"{REPORTS}/feature_importance.png", dpi=120); plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run()
