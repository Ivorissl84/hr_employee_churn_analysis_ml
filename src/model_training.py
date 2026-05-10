import os
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.inspection import permutation_importance
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


def train_model(X_train, y_train, X_test, y_test, model_path="model/random_forest_model.pkl"):
    """
    Trainiert ein Balanced Random Forest Modell zur Vorhersage von Mitarbeiter-Churn.
    Erweiterte Version:
    - Modelltraining
    - Klassische Metriken
    - Confusion Matrix (Plot)
    - Gini Importance
    - Permutation Importance
    - Gemeinsame Tabelle (Gini vs. Permutation)
    - Gemeinsames Kurvendiagramm (Feature Stability)
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

    # ---------------------------------------------------
    # 4. Evaluation der Modellleistung
    # ---------------------------------------------------
    print("\n--- Modellbewertung ---")
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, zero_division=0))

    # ---------------------------------------------------
    # 5. Confusion Matrix
    # ---------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    print("\n--- Confusion Matrix ---")
    print(cm)

    # Confusion Matrix Plot
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
    plt.show()
    plt.close()

    print(f"Confusion Matrix Plot gespeichert unter: {cm_path}")

    # ---------------------------------------------------
    # 6. Gini Feature Importance
    # ---------------------------------------------------
    gini_df = pd.DataFrame({
        "feature": X_train.columns,
        "gini_importance": model.feature_importances_
    }).sort_values(by="gini_importance", ascending=False)

    print("\n--- Gini Feature Importance ---")
    print(gini_df)

    # ---------------------------------------------------
    # 7. Permutation Importance
    # ---------------------------------------------------
    print("\n--- Permutation Importance (Testdaten) ---")

    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    perm_df = pd.DataFrame({
        "feature": X_test.columns,
        "perm_importance": perm.importances_mean
    }).sort_values(by="perm_importance", ascending=False)

    print(perm_df)

    # ---------------------------------------------------
    # 8. Gemeinsame Tabelle (Gini vs. Permutation)
    # ---------------------------------------------------
    merged = gini_df.merge(perm_df, on="feature", how="inner")

    print("\n--- Vergleich: Gini vs. Permutation Importance ---")
    print(merged)

    # ---------------------------------------------------
    # 9. Gemeinsames Kurvendiagramm (Feature Stability)
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
    plt.title("Feature Stability: Gini vs. Permutation Importance")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.legend()

    stability_path = os.path.join(MODEL_PLOT_DIR, "feature_stability.png")
    plt.savefig(stability_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Feature Stability Plot gespeichert unter: {stability_path}")

    # ---------------------------------------------------
    # 10. Automatische Interpretation
    # ---------------------------------------------------
    print("\n--- Automatische Interpretation der Feature-Stabilität ---")

    for _, row in merged.iterrows():
        diff = abs(row["gini_importance"] - row["perm_importance"])

        if diff < 0.01:
            print(f"{row['feature']}: Sehr stabil ✓")
        elif diff < 0.05:
            print(f"{row['feature']}: Moderat stabil ~")
        else:
            print(f"{row['feature']}: Instabil / mögliches Artefakt ✗")

    # ---------------------------------------------------
    # 11. Modell speichern
    # ---------------------------------------------------
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)

    joblib.dump(model, model_path)
    print(f"\nModell gespeichert unter: {model_path}")

    return model, merged
