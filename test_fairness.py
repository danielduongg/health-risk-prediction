import os
from generate_data import generate
import train as T, fairness as F

def test_mitigation_reduces_recall_gap(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data"); os.makedirs("reports", exist_ok=True)
    generate(n=2500, seed=4).to_csv("data/diabetes.csv", index=False)
    T.run()
    s = F.run()
    assert "group_fairness" in s
    # per-group thresholds should not widen the recall gap
    assert s["recall_gap_after"] <= s["recall_gap_before"] + 1e-9
