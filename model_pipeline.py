import pandas as pd
import joblib
import mysql.connector
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_data_from_mysql():
    """Charge les données depuis MySQL"""
    db_config = {
        "host": "localhost",
        "user": "root",  # Remplacez par l'utilisateur correct
        "password": "root",  # Remplacez par le mot de passe correct
        "database": "churn_db"
    }

    try:
        # Connexion à la base de données
        conn = mysql.connector.connect(**db_config)
        
        # Exécution de la requête SQL
        query = "SELECT * FROM churn_data"
        df = pd.read_sql(query, conn)
        
        # Fermeture de la connexion
        conn.close()
        
        return df
    
    except mysql.connector.Error as err:
        print(f"Erreur de connexion MySQL: {err}")
        return None
    
    except Exception as e:
        print(f"Une erreur est survenue: {e}")
        return None
def prepare_data(data):
    """Prépare les données en encodant les variables catégorielles et en normalisant les features."""
    
    if 'Churn' not in data.columns:
        raise ValueError("La colonne 'Churn' est absente du dataset.")

    label_encoders = {}
    
    # Encodage des variables catégorielles
    categorical_columns = ['State', 'International_plan', 'Voice_mail_plan']
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Séparation des features et de la cible
    X = data.drop(columns=['Churn'])
    y = data['Churn'].astype(int)  # Convertir en binaire (0 ou 1)

    # Normalisation des features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Séparation en train/test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler, label_encoders

def train_model(X_train, y_train):
    """Entraîne un modèle de réseau de neurones MLP."""
    model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Évalue le modèle sur l'ensemble de test."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

def save_model(model, scaler, label_encoders, filename='model.joblib'):
    """Sauvegarde le modèle et les transformateurs."""
    joblib.dump({'model': model, 'scaler': scaler, 'encoders': label_encoders}, filename)

def load_model(filename='model.joblib'):
    """Charge un modèle sauvegardé."""
    return joblib.load(filename)
