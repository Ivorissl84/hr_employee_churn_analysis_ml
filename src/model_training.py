import os
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
from sklearn.inspection import permutation_importance
import joblib


def train_model(X_train, y_train, X_test, y_test, model_path="model/random_forest_model.pkl"):
    """
    Trainiert ein Balanced Random Forest Modell und gibt ALLE relevanten
    Modellinformationen als Dictionary zurück – ohne Prints, ohne Plots.

    Rückgabe enthält:
    - Modellobjekt
    - Klassische Metriken
    - Confusion Matrix
    - ROC-Daten
    - Precision-Recall-Daten
    - Gini Importance
    - Permutation Importance (Top-20)
    - Vergleichstabelle (Gini vs. Permutation)
    """

    # ---------------------------------------------------
    # 1. Modell definieren
    # ---------------------------------------------------
    model = BalancedRandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=3,
        sampling_strategy="auto",
        random_state=42
    )

    # ---------------------------------------------------
    # 2. Modell trainieren
    # ---------------------------------------------------
    model.fit(X_train, y_train)

    # ---------------------------------------------------
    # 3. Vorhersagen
    # ---------------------------------------------------
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # ---------------------------------------------------
    # 4. Klassische Metriken
    # ---------------------------------------------------
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "classification_report": classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    }

    # ---------------------------------------------------
    # 5. Confusion Matrix
    # ---------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)

    # ---------------------------------------------------
    # 6. ROC-Kurve
    # ---------------------------------------------------
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    roc_data = {
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc
    }

    # ---------------------------------------------------
    # 7. Precision-Recall-Kurve
    # ---------------------------------------------------
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)

    pr_data = {
        "precision": precision_vals,
        "recall": recall_vals
    }

    # ---------------------------------------------------
    # 8. Gini Importance
    # ---------------------------------------------------
    gini_df = pd.DataFrame({
        "feature": X_train.columns,
        "gini_importance": model.feature_importances_
    }).sort_values(by="gini_importance", ascending=False)

    # ---------------------------------------------------
    # 9. Permutation Importance (Top-20)
    # ---------------------------------------------------
    top_features = gini_df.head(20)["feature"].tolist()

    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=3,
        random_state=42,
        n_jobs=-1
    )

    perm_df = pd.DataFrame({
        "feature": X_test.columns,
        "perm_importance": perm.importances_mean
    })

    perm_df = perm_df[perm_df["feature"].isin(top_features)]
    perm_df = perm_df.sort_values(by="perm_importance", ascending=False)

    # ---------------------------------------------------
    # 10. Vergleichstabelle
    # ---------------------------------------------------
    merged = gini_df.merge(perm_df, on="feature", how="inner")

    # ---------------------------------------------------
    # 11. Modell speichern
    # ---------------------------------------------------
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)

    joblib.dump(model, model_path)

    # ---------------------------------------------------
    # 12. Rückgabe
    # ---------------------------------------------------
    return {
        "model": model,
        "metrics": metrics,
        "confusion_matrix": cm,
        "roc": roc_data,
        "pr": pr_data,
        "gini_importance": gini_df,
        "permutation_importance": perm_df,
        "stability_table": merged,
        "model_path": model_path
    }
