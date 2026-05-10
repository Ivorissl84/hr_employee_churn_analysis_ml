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


def visualize_feature_engineering(X_train):
    """
    Minimalistische Feature-Engineering-Visualisierung:
    - EIN Plot: Anzahl Features nach Typ
    - EIN Report: Übersicht über die erzeugten Features
    """

    print("Starte reduzierte Feature-Engineering-Visualisierung...")

    # ---------------------------------------------------
    # 1. Feature-Typen bestimmen
    # ---------------------------------------------------
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    onehot_cols = [col for col in X_train.columns if "_" in col and col not in numeric_cols]
    other_cols = [col for col in X_train.columns if col not in numeric_cols + onehot_cols]

    # ---------------------------------------------------
    # 2. EIN Balkendiagramm: Anzahl Features nach Typ
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
    plt.show()
    plt.close()

    # ---------------------------------------------------
    # 3. Minimalistischer Feature-Engineering-Report
    # ---------------------------------------------------
    print("\n--- Feature-Engineering-Report ---")
    print(f"Gesamtzahl Features: {len(X_train.columns)}")
    print(f"Numerische Features: {len(numeric_cols)}")
    print(f"One-Hot-Features:   {len(onehot_cols)}")
    print(f"Sonstige Features:  {len(other_cols)}")

    if len(onehot_cols) > 30:
        print("⚠️ Warnung: Sehr viele One-Hot-Features – mögliche Überdimensionierung.")

    if len(numeric_cols) == 0:
        print("⚠️ Warnung: Keine numerischen Features gefunden.")

    print("Feature-Engineering-Visualisierung abgeschlossen.")
