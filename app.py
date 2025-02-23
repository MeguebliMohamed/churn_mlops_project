from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

# Création de l'application FastAPI
app = FastAPI(title="API de Prédiction avec FastAPI", version="1.0")

# Charger le modèle et le scaler
model_data = joblib.load("model.pkl")  # Assurez-vous d'avoir ce fichier dans votre répertoire
model = model_data["model"]
scaler = model_data["scaler"]

# Définir un modèle de données pour l'entrée utilisateur
class FeaturesInput(BaseModel):
    features: list[float]  # Une liste de valeurs numériques

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de prédiction !"}

@app.post("/predict/")
def predict(input_data: FeaturesInput):
    try:
        # Transformation des données d'entrée
        features_array = np.array(input_data.features).reshape(1, -1)
        scaled_features = scaler.transform(features_array)
        
        # Faire une prédiction
        prediction = model.predict(scaled_features)
        
        return {"prediction": prediction.tolist()}
    
    except Exception as e:
        return {"error": str(e)}

