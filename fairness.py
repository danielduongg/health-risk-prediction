"""Subgroup fairness audit + a mitigation.

Audit: per age-band recall (sensitivity), FPR, precision, selection rate, and
aggregate group-fairness metrics (demographic-parity gap, equalized-odds gaps).

Mitigation: pick a *per-group* decision threshold that targets equal recall, and
show the recall gap collapse. For a screening tool, equal recall ('equal
opportunity') is the fairness criterion that matters — equal chance of being
caught if you truly have the disease.
"""
from __future__ import annotations
import json, logging, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger("health.fairness")
REPORTS = "reports"
TARGET_RECALL = 0.70


def _rates(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return dict(
        recall=round(tp / (tp + fn), 3) if (tp + fn) else float("nan"),
        fpr=round(fp / (fp + tn), 3) if (fp + tn) else float("nan"),
        precision=round(tp / (tp + fp), 3) if (tp + fp) else float("nan"),
        selection_rate=round(float(pred.mean()), 3))


def _threshold_for_recall(scores, y, target):
    """Smallest threshold whose recall >= target within this group."""
    pos = np.sort(scores[y == 1])[::-1]
    if len(pos) == 0:
        return 0.5
    k = max(1, int(np.ceil(target * len(pos))))
    return float(pos[min(k - 1, len(pos) - 1)])


def run(preds_csv=f"{REPORTS}/test_predictions.csv", score_col="risk_logreg"):
    os.makedirs(REPORTS, exist_ok=True)
    df = pd.read_csv(preds_csv)
    df["age_band"] = pd.cut(df["Age"], [20, 30, 45, 90], labels=["21-30", "31-45", "46+"])
    y = df["Outcome"].values; s = df[score_col].values

    # ---- audit at a single global 0.5 threshold ----
    base_rows, recall_before = [], {}
    for band, sub in df.groupby("age_band", observed=True):
        yy = sub["Outcome"].values; pred = (sub[score_col].values >= 0.5).astype(int)
        r = _rates(yy, pred); r.update(age_band=band, n=len(sub),
                                       prevalence=round(float(yy.mean()), 3))
        base_rows.append(r); recall_before[band] = r["recall"]
    audit = pd.DataFrame(base_rows)[["age_band","n","prevalence","recall","fpr","precision","selection_rate"]]

    def gap(col): return round(float(np.nanmax(audit[col]) - np.nanmin(audit[col])), 3)
    group_fairness = dict(demographic_parity_gap=gap("selection_rate"),
                          equalized_odds_tpr_gap=gap("recall"),
                          equalized_odds_fpr_gap=gap("fpr"))

    # ---- mitigation: per-group thresholds targeting equal recall ----
    recall_after, thresholds = {}, {}
    for band, sub in df.groupby("age_band", observed=True):
        yy = sub["Outcome"].values; sc = sub[score_col].values
        thr = _threshold_for_recall(sc, yy, TARGET_RECALL)
        thresholds[band] = round(thr, 3)
        recall_after[band] = _rates(yy, (sc >= thr).astype(int))["recall"]
    gap_before = round(max(recall_before.values()) - min(recall_before.values()), 3)
    gap_after = round(max(recall_after.values()) - min(recall_after.values()), 3)

    summary = dict(audit=audit.to_dict("records"), group_fairness=group_fairness,
                   recall_gap_before=gap_before, recall_gap_after=gap_after,
                   per_group_thresholds=thresholds, target_recall=TARGET_RECALL)
    with open(f"{REPORTS}/fairness_audit.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    audit.to_csv(f"{REPORTS}/fairness_audit.csv", index=False)

    bands = list(recall_before.keys()); x = np.arange(len(bands)); w = 0.35
    plt.figure(figsize=(7.5, 4.5))
    plt.bar(x - w/2, [recall_before[b] for b in bands], w, label="Before (global 0.5)", color="#c05621")
    plt.bar(x + w/2, [recall_after[b] for b in bands], w, label="After (per-group threshold)", color="#2f855a")
    plt.axhline(TARGET_RECALL, ls="--", c="k", alpha=.4, label=f"target recall {TARGET_RECALL}")
    plt.xticks(x, bands); plt.ylim(0, 1); plt.ylabel("Recall (sensitivity)")
    plt.title(f"Recall by age band — gap {gap_before} → {gap_after} after mitigation")
    plt.legend(); plt.tight_layout(); plt.savefig(f"{REPORTS}/fairness_mitigation.png", dpi=120); plt.close()

    print("=== Audit (global 0.5 threshold) ===")
    print(audit.to_string(index=False))
    print("\nGroup fairness:", group_fairness)
    print(f"Recall gap: {gap_before} (before) -> {gap_after} (after per-group thresholds)")
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run()
