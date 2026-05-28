from src.data_generator import generate_payment_data
from src.preprocessing import BASELINE_FEATURES, IMPROVED_FEATURES, TARGET_COLUMN, split_features_target


def test_data_generator_shape_and_target():
    df = generate_payment_data(n_samples=100, random_state=7)
    assert len(df) == 100
    assert TARGET_COLUMN in df.columns
    assert set(df[TARGET_COLUMN].unique()).issubset({0, 1})


def test_feature_sets_exist_in_generated_data():
    df = generate_payment_data(n_samples=50, random_state=7)
    for column in BASELINE_FEATURES + IMPROVED_FEATURES:
        assert column in df.columns


def test_split_features_target_returns_expected_shapes():
    df = generate_payment_data(n_samples=40, random_state=7)
    X, y = split_features_target(df, BASELINE_FEATURES)
    assert X.shape == (40, len(BASELINE_FEATURES))
    assert y.shape[0] == 40
