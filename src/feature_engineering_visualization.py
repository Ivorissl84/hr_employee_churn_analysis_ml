import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------
# Ordner für Feature-Engineering-Plots
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FE_PLOT_DIR = os.path.join(BASE_DIR, "plots/feature_engineering")


def save_plot(name):
    """Speichert den aktuellen Plot im feature_engineering/-Ordner."""
    os.makedirs(FE_PLOT_DIR, exist_ok=True)
    path = os.path.join(FE_PLOT_DIR, f"{name}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Plot gespeichert: {path}")


def visualize_feature_engineering(X_train, feature_importances=None, permutation_importances=None):
    """
    Erweiterte Feature-Engineering-Visualisierung:
    - Anzahl Features nach Typ
    - Feature Importance (Top 10)
    - Vergleich: Gini vs. Permutation Importance
    - Korrelation der encoded Features
    """

    print("Starte erweiterte Feature-Engineering-Visualisierung...")

    # ---------------------------------------------------
    # 1. Feature-Typen bestimmen
    # ---------------------------------------------------
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    onehot_cols = [col for col in X_train.columns if "_" in col and col not in numeric_cols]
    other_cols = [col for col in X_train.columns if col not in numeric_cols + onehot_cols]

    # ---------------------------------------------------
    # 2. Plot: Anzahl Features nach Typ
    # ---------------------------------------------------
    plt.figure(figsize=(6, 4))
    counts = {
        "Numerisch": len(numeric_cols),
        "One-Hot": len(onehot_cols),
        "Sonstige": len(other_cols)
    }
    sns.barplot(x=list(counts.keys()), y=list(counts.values()))
    plt.title("Feature-Typen nach Feature Engineering")
    plt.ylabel("Anzahl Features")
    save_plot("feature_type_counts")
    plt.close()

    # ---------------------------------------------------
    # 3. Feature Importance (Top 10)
    # ---------------------------------------------------
    if feature_importances is not None:
        top10 = feature_importances.sort_values("gini_importance", ascending=False).head(10)

        plt.figure(figsize=(8, 5))
        sns.barplot(
            x="gini_importance",
            y="feature",
            data=top10,
            hue="feature",
            palette="viridis",
            legend=False
        )

        plt.title("Top 10 Feature Importances (Gini)")
        plt.xlabel("Gini Importance")
        plt.ylabel("Feature")
        save_plot("feature_importance_top10")
        plt.close()

    # ---------------------------------------------------
    # 4. Vergleich: Gini vs. Permutation Importance
    # ---------------------------------------------------
    if feature_importances is not None and permutation_importances is not None:
        merged = feature_importances.merge(permutation_importances, on="feature", how="left")

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x="gini_importance",
            y="perm_importance",
            data=merged
        )
        plt.title("Gini vs. Permutation Importance")
        plt.xlabel("Gini Importance")
        plt.ylabel("Permutation Importance")
        save_plot("feature_importance_comparison")
        plt.close()

    # ---------------------------------------------------
    # 5. Korrelation der encoded Features
    # ---------------------------------------------------
    plt.figure(figsize=(12, 10))
    corr = X_train.corr()
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Feature-Korrelationen nach Encoding")
    save_plot("feature_correlation_encoded")
    plt.close()

    # ---------------------------------------------------
    # 6. Report
    # ---------------------------------------------------
    print("\n--- Feature-Engineering-Report ---")
    print(f"Gesamtzahl Features: {len(X_train.columns)}")
    print(f"Numerische Features: {len(numeric_cols)}")
    print(f"One-Hot-Features:   {len(onehot_cols)}")
    print(f"Sonstige Features:  {len(other_cols)}")

    if len(onehot_cols) > 30:
        print("⚠️ Warnung: Sehr viele One-Hot-Features – mögliche Überdimensionierung.")

    print("Feature-Engineering-Visualisierung abgeschlossen.")
