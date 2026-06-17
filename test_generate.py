from generate_data import generate

def test_columns_and_positive_rate():
    df = generate(n=1500, seed=1)
    assert df["Outcome"].isin([0, 1]).all()
    assert 0.28 < df["Outcome"].mean() < 0.42

def test_impossible_zeros_injected():
    df = generate(n=2000, seed=2)
    for col in ["Insulin", "SkinThickness"]:
        assert (df[col] == 0).sum() > 0   # the dataset's signature missingness
