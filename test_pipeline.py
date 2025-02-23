import pytest
import mysql.connector
import pandas as pd  # Ensure pandas is imported
import numpy as np
from model_pipeline import prepare_data, train_model, evaluate_model, load_data_from_mysql

# Configuration de la connexion MySQL pour les tests
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "root",  # Use your actual password here
    "database": "churn_db"
}

@pytest.fixture
def sample_data():
    """Charge un échantillon de données depuis MySQL pour les tests."""
    conn = mysql.connector.connect(**db_config)
    query = "SELECT * FROM churn_data LIMIT 100"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# ✅ Test Unitaire : Vérification du chargement des données
def test_load_data(sample_data):
    assert sample_data is not None
    assert not sample_data.empty
    assert 'Churn' in sample_data.columns

# ✅ Test Unitaire : Vérification du Preprocessing
def test_prepare_data(sample_data):
    X_train, X_test, y_train, y_test, _, _ = prepare_data(sample_data)  # If it takes args, pass them here
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert y_train.shape[0] > 0
    assert y_test.shape[0] > 0

# ✅ Test Unitaire : Entraînement du modèle
def test_train_model():
    X_train = np.array([[1,128,415,0,1,25,265.1,110,45.07,197.4,99,16.78,244.7,91,11.01,10,3,2.7,1]])
    y_train = np.array([0])  # 0 = Pas de churn

    model = train_model(X_train, y_train)

    assert model is not None
    assert hasattr(model, "predict")

# ✅ Test Fonctionnel : Évaluation du modèle
def test_evaluate_model(sample_data):
    X_train, X_test, y_train, y_test, _, _ = prepare_data(sample_data)  # Make sure prepare_data() is setup correctly
    model = train_model(X_train, y_train)

    accuracy, _ = evaluate_model(model, X_test, y_test)  # Assuming it returns a tuple, extract accuracy

    assert accuracy > 0  # La précision doit être supérieure à 0
