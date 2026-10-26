import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

RACINE = Path(__file__).resolve().parent.parent
os.chdir(RACINE)
print("Dossier courant :", os.getcwd())

with open("params.yaml") as f:
    params = yaml.safe_load(f)["train"]

df = pd.read_csv("data/etudiants.csv")
X = df[["heures_etude", "heures_sommeil"]]
y = df["reussi"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=params["test_size"],
    random_state=params["random_state"],
    stratify=y,
)

# entraîner + enregistrer avec MLflow
mlflow.set_experiment("reussite_etudiants")   # crée/choisit un "dossier" d'expériences
mlflow.set_tracking_uri("sqlite:///mlflow.db")
with mlflow.start_run():                      # un "run" = un essai
    model = LogisticRegression(C=params["C"])
    model.fit(X_train, y_train)               # <-- l'apprentissage se fait ici

    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions) # % de bonnes réponses

    mlflow.log_params(params)                 # MLflow note les réglages
    mlflow.log_metric("accuracy", acc)        # MLflow note le score
    mlflow.sklearn.log_model(model, name="modele")  # MLflow garde une copie du modèle

# ---------- 5. Sauvegarder les résultats pour DVC ----------
os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/model.pkl")        # le modèle entraîné

with open("../metrics.json", "w") as f:          # le score, lisible par DVC
    json.dump({"accuracy": acc}, f)

print("Accuracy :", acc)

# ---------- 6. Tester une prédiction ----------
nouvel_etudiant = pd.DataFrame([[6, 7]], columns=X.columns)  # 6 h d'étude, 7 h de sommeil
print("Prédiction pour 6h d'étude / 7h de sommeil :", model.predict(nouvel_etudiant)[0])
