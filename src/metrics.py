from __future__ import annotations
from typing import Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """
    Calcule les métriques de performance pour la classification des troubles du sommeil.
    Utilise la moyenne 'weighted' pour balancer la précision et le rappel sur les 3 classes.
    """

    # Calcul des métriques standards
    # On convertit explicitement en float pour respecter les contraintes du dictionnaire
    
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }
