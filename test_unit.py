import pytest
import pandas as pd
from model_pipeline import load_data, prepare_data, train_model, evaluate_model, save_model, load_model
import joblib
import os

# Mock data pour les tests
@pytest.fixture
def sample_data():
    """Génère un DataFrame de données factices pour les tests."""
    data = pd.DataFrame({
        'State': ['CA', 'TX', 'NY', 'FL', 'OH'],
        'Account length': [107, 137, 84, 132, 94],
        'Area code': [408, 415, 510, 408, 415],
        'International plan': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'Voice mail plan': ['No', 'Yes', 'No', 'Yes', 'No'],
        'Number vmail messages': [0, 37, 0, 12, 0],
        'Total day minutes': [189.7, 295.4, 155.3, 268.8, 187.2],
        'Total day calls': [123, 109, 98, 120, 101],
        'Total day charge': [32.25, 50.22, 26.40, 45.70, 31.82],
        'Total eve minutes': [213.2, 194.3, 245.3, 172.5, 233.1],
        'Total eve calls': [98, 87, 91, 103, 92],
        'Total eve charge': [18.12, 16.52, 20.85, 14.66, 19.81],
        'Total night minutes': [224.5, 232.9, 256.6, 221.3, 218.9],
        'Total night calls': [112, 101, 99, 88, 97],
        'Total night charge': [10.10, 10.48, 11.55, 9.96, 9.85],
        'Total intl minutes': [12, 8, 15, 7, 13],
        'Total intl calls': [4, 3, 5, 2, 6],
        'Total intl charge': [3.24, 2.16, 4.05, 1.89, 3.51],
        'Customer service calls': [2, 1, 3, 0, 2],
        'Churn': [1, 0, 1, 0, 1]  # 1 = Churn, 0 = No Churn
    })
    return data
def test_prepare_data(sample_data):
    X_train, X_test, y_train, y_test, scaler, label_encoders = prepare_data(sample_data)
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(y_train) > 0
    assert len(y_test) > 0

def test_train_model(sample_data):
    X_train, X_test, y_train, y_test, _, _ = prepare_data(sample_data)
    model = train_model(X_train, y_train)
    assert model is not None

def test_evaluate_model(sample_data):
    X_train, X_test, y_train, y_test, _, _ = prepare_data(sample_data)
    model = train_model(X_train, y_train)
    accuracy, report = evaluate_model(model, X_test, y_test)
    assert accuracy is not None
    assert isinstance(report, str)

def test_save_and_load_model(sample_data, tmp_path):
    model_path = tmp_path / "model.joblib"
    X_train, X_test, y_train, y_test, scaler, label_encoders = prepare_data(sample_data)
    model = train_model(X_train, y_train)

    save_model(model, scaler, label_encoders, str(model_path))
    loaded_data = load_model(str(model_path))

    assert "model" in loaded_data
    assert "scaler" in loaded_data
    assert "encoders" in loaded_data
