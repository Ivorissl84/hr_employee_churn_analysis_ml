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
import matplotlib.pyplot as plt
import seaborn as sns


def train_model(X_train, y_train, X_test, y_test, model_path="model/random_forest_model.pkl"):
    """
    Trainiert ein Balanced Random Forest Modell zur Vorhersage von Mitarbeiter-Churn.
    Enthält:
    - Modelltraining
    - Klassische Metriken
    - Confusion Matrix (Plot)
    - ROC-Kurve
    - Precision-Recall-Kurve
    - Gini Importance
    - Permutation Importance (nur Top-20 Features)
    - Gemeinsame Tabelle (Gini vs. Permutation)
    - Feature Stability Plot
    """

    print("Starte Modelltraining...")

    # ---------------------------------------------------
    # Ordner für Modellplots
    # ---------------------------------------------------
    MODEL_PLOT_DIR = "plots/model"
    os.makedirs(MODEL_PLOT_DIR, exist_ok=True)

    # ---------------------------------------------------
    # 1. Balanced Random Forest definieren
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
    # 3. Vorhersagen auf Testdaten
    # ---------------------------------------------------
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # ---------------------------------------------------
    # 4. Evaluation der Modellleistung
    # ---------------------------------------------------
    print("\n=== MODEL METRICS ===")

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    # ---------------------------------------------------
    # 5. Confusion Matrix
    # ---------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== CONFUSION MATRIX ===")
    print(cm)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greys",
        cbar=False,
        linewidths=0.5,
        linecolor="black"
    )
    plt.title("Confusion Matrix (Balanced Random Forest)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    cm_path = os.path.join(MODEL_PLOT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Confusion Matrix Plot gespeichert unter: {cm_path}")

    # ---------------------------------------------------
    # 6. ROC-Kurve
    # ---------------------------------------------------
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", color="blue")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.title("ROC-Kurve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    roc_path = os.path.join(MODEL_PLOT_DIR, "roc_curve.png")
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"ROC-Kurve gespeichert unter: {roc_path}")

    # ---------------------------------------------------
    # 7. Precision-Recall-Kurve
    # ---------------------------------------------------
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="green")
    plt.title("Precision-Recall-Kurve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    pr_path = os.path.join(MODEL_PLOT_DIR, "precision_recall_curve.png")
    plt.savefig(pr_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Precision-Recall-Kurve gespeichert unter: {pr_path}")

    # ---------------------------------------------------
    # 8. Gini Feature Importance
    # ---------------------------------------------------
    gini_df = pd.DataFrame({
        "feature": X_train.columns,
        "gini_importance": model.feature_importances_
    }).sort_values(by="gini_importance", ascending=False)

    print("\n=== GINI FEATURE IMPORTANCE ===")
    print(gini_df)

    # ---------------------------------------------------
    # 9. Permutation Importance (Top-20, aber stabil)
    # ---------------------------------------------------
    print("\n=== PERMUTATION IMPORTANCE (TESTDATEN, TOP-20 FEATURES) ===")

    # Top-20 Features nach Gini
    top_features = gini_df.head(20)["feature"].tolist()

    # WICHTIG:
    # Modell bekommt ALLE Features → kein Fehler, kein Hänger
    perm = permutation_importance(
        model,
        X_test,  # komplette Matrix!
        y_test,
        n_repeats=3,  # schnell & stabil
        random_state=42,
        n_jobs=-1
    )

    # Permutation Importance für ALLE Features → jetzt filtern wir die Top-20 heraus
    perm_df = pd.DataFrame({
        "feature": X_test.columns,
        "perm_importance": perm.importances_mean
    })

    # Nur Top-20 behalten
    perm_df = perm_df[perm_df["feature"].isin(top_features)]
    perm_df = perm_df.sort_values(by="perm_importance", ascending=False)

    print(perm_df)

    # ---------------------------------------------------
    # 10. Gemeinsame Tabelle (Gini vs. Permutation)
    # ---------------------------------------------------
    merged = gini_df.merge(perm_df, on="feature", how="inner")

    print("\n=== VERGLEICH: GINI vs. PERMUTATION IMPORTANCE (TOP-20) ===")
    print(merged)

    # ---------------------------------------------------
    # 11. Feature Stability Plot
    # ---------------------------------------------------
    plt.figure(figsize=(12, 6))

    plt.plot(
        merged["feature"],
        merged["gini_importance"],
        marker="o",
        label="Gini Importance",
        color="blue"
    )

    plt.plot(
        merged["feature"],
        merged["perm_importance"],
        marker="o",
        label="Permutation Importance",
        color="orange"
    )

    plt.xticks(rotation=45)
    plt.title("Feature Stability: Gini vs. Permutation Importance (Top-20)")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.legend()

    stability_path = os.path.join(MODEL_PLOT_DIR, "feature_stability.png")
    plt.savefig(stability_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Feature Stability Plot gespeichert unter: {stability_path}")

    # ---------------------------------------------------
    # 12. Modell speichern
    # ---------------------------------------------------
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)

    joblib.dump(model, model_path)
    print(f"\nModell gespeichert unter: {model_path}")

    # ---------------------------------------------------
    # 13. Rückgabe für Feature-Engineering-Visualisierung
    # ---------------------------------------------------
    return model, merged
