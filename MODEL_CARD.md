# Model Card — Diabetes Risk Prediction

## Intended use
Educational demonstration of a responsible screening-model workflow. **Not a
medical device; not for clinical decisions.**

## Data
Synthetic, reproducing the statistical structure of the Pima Indians Diabetes
dataset (8 clinical features; ~35% positive), including its known
physiologically-impossible zeros, which are encoded as missing and imputed.

## Model
Calibrated logistic regression (calibration method selected by Brier score) and
a tuned XGBoost, with median imputation and standardization. Model selection via
5-fold cross-validated AUC on the training split only.

## Metrics
Reported on a 25% stratified holdout: ROC-AUC, PR-AUC, recall (sensitivity),
precision, Brier score. See `reports/metrics.json`.

## Fairness
Audited by age band for recall, FPR, precision and selection rate, plus
aggregate demographic-parity and equalized-odds gaps. A per-group threshold
mitigation targets equal recall ("equal opportunity"); see
`reports/fairness_audit.json` and `reports/fairness_mitigation.png`.

## Ethical considerations & limitations
- Screening favors **recall** — a missed case is costlier than a false alarm —
  but more alarms mean more follow-up cost and patient anxiety.
- Subgroup performance varies; a single global threshold under-serves
  low-prevalence groups. Always evaluate and tune per subgroup.
- Synthetic data cannot capture real clinical complexity; a deployed tool needs
  prospective validation, calibration monitoring, and clinician oversight.
