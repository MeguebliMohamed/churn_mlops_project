# ==============================
# Déclaration des variables
# ==============================
PYTHON=python3
ENV_NAME=venv
REQUIREMENTS=requirements.txt
DATA_FILE=churn-data.csv
MODEL_PATH=model.joblib

# ==============================
# 1. Configuration de l'environnement
# ==============================
.PHONY: setup
setup:
	@echo "🔧 Création de l'environnement virtuel et installation des dépendances..."
	@$(PYTHON) -m venv $(ENV_NAME)
	@. $(ENV_NAME)/bin/activate && pip install -r $(REQUIREMENTS)

# ==============================
# 2. Vérification et formatage du code
# ==============================
.PHONY: lint format security
lint:
	@echo "🔍 Vérification du style du code avec flake8..."
	@. $(ENV_NAME)/bin/activate && flake8 .

format:
	@echo "📝 Formatage automatique du code avec black..."
	@. $(ENV_NAME)/bin/activate && black .

security:
	@echo "🔒 Vérification des vulnérabilités du code avec bandit..."
	@. $(ENV_NAME)/bin/activate && bandit -r .

quality: lint format security
# chargement base de donne et upload server
.PHONY: mysql_up mysql_load_data

mysql_up:
	@echo "🚀 Démarrage de MySQL avec Docker..."
	@docker-compose up -d

# ==============================
# 3. Chargement et préparation des données
# ==============================
.PHONY: data
data:
	@echo "🔄 Préparation des données (train/test split)..."
	@. $(ENV_NAME)/bin/activate && $(PYTHON) main.py prepare --data $(DATA_FILE)

# ==============================
# 4. Entraînement du modèle
# ==============================
.PHONY: train
train: data
	@echo "🤖 Entraînement du modèle..."
	@. $(ENV_NAME)/bin/activate && $(PYTHON) main.py train --data $(DATA_FILE)

# ==============================
# 5. Évaluation du modèle
# ==============================
.PHONY: evaluate
evaluate: train
	@echo "📊 Évaluation du modèle..."
	@. $(ENV_NAME)/bin/activate && $(PYTHON) main.py evaluate --data $(DATA_FILE)

# ==============================
# 6. Sauvegarde du modèle
# ==============================
.PHONY: save
save: train
	@echo "💾 Sauvegarde du modèle..."
	@. $(ENV_NAME)/bin/activate && $(PYTHON) main.py save --data $(DATA_FILE) --save $(MODEL_PATH)

# ==============================
# 7. Chargement d'un modèle existant
# ==============================
.PHONY: load
load:
	@echo "📥 Chargement du modèle..."
	@. $(ENV_NAME)/bin/activate && $(PYTHON) main.py load --load $(MODEL_PATH)

# ==============================
# 8. Exécution des tests
# ==============================
.PHONY: test_unit test_pipeline

test_unit:
	@echo "🧪 Exécution des tests unitaires..."
	@. $(ENV_NAME)/bin/activate && pytest test_pipeline.py -k "test_load_data or test_prepare_data or test_train_model"

test_pipeline:
	@echo "🛠 Exécution des tests fonctionnels..."
	@. $(ENV_NAME)/bin/activate && pytest test_pipeline.py -k "test_evaluate_model"

# ==============================
# 9. Démarrage du serveur Jupyter Notebook
# ==============================
.PHONY: notebook
notebook:
	@echo "📔 Démarrage de Jupyter Notebook..."
	@. $(ENV_NAME)/bin/activate && jupyter notebook

# ==============================
# 10. Nettoyage des fichiers temporaires
# ==============================
.PHONY: clean
clean:
	@echo "🧹 Nettoyage des fichiers inutiles..."
	@rm -rf __pycache__ .pytest_cache .mypy_cache $(ENV_NAME)
	@find . -type d -name "*.egg-info" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "*.ipynb_checkpoints" -exec rm -rf {} +
