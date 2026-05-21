"""Helpers for saving evaluation results."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from config import MODEL_METRICS_FILE
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# 1. Chemins des modèles (basés sur vos informations)
RF_MODEL_PATH = r"models\random_forest_sleep_20260506_1723.pkl"
XGB_MODEL_PATH = r"models\xgboost_sleep_20260506_1723.pkl"

from metrics import compute_metrics

def compare_models(X_test, y_test):
    """Charge les modèles, prédit et affiche la comparaison."""
    
    # Chargement des modèles
    rf_model = joblib.load(RF_MODEL_PATH)
    xgb_model = joblib.load(XGB_MODEL_PATH)
    
    # Prédictions
    y_pred_rf = rf_model.predict(X_test)
    y_pred_xgb = xgb_model.predict(X_test)
    
    # Calcul des métriques
    metrics_rf = compute_metrics(y_test, y_pred_rf, "Random Forest")
    metrics_xgb = compute_metrics(y_test, y_pred_xgb, "XGBoost")
    
    # Création du DataFrame de comparaison
    df_comparison = pd.read_csv(r"data\Sleep_health_and_lifestyle_dataset.csv")
    
    # Mise en forme pour faciliter la lecture (Transposition)
    df_styled = df_comparison.set_index("Model").T
    
    # Ajout d'une colonne Delta pour voir qui gagne et de combien
    df_styled['Diff (XGB - RF)'] = df_styled['XGBoost'] - df_styled['Random Forest']
    
    df_styled.to_csv('MODEL_METRICS_FILE.csv',index=True)
    return df_styled


# --- EXÉCUTION ---
# Remarque : Remplacez X_test et y_test par vos données de validation
# results = compare_models(X_test, y_test)
# print(results)