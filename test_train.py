import os
from generate_data import generate
import train as T

def test_train_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data"); os.makedirs("reports", exist_ok=True)
    generate(n=1800, seed=3).to_csv("data/diabetes.csv", index=False)
    s = T.run()
    assert s["holdout"]["logistic_regression"]["roc_auc"] > 0.7
    assert os.path.exists("reports/test_predictions.csv")
    assert os.path.exists("reports/model.joblib")
