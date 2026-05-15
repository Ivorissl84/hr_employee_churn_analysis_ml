import pandas as pd


def visualize_feature_engineering(X_train, gini_importances=None, perm_importances=None):
    """
    Bereitet Feature-Engineering-Visualisierungsdaten vor – ohne Prints, ohne Plots.
    Gibt zurück:
    - feature_summary: Anzahl Features nach Typ
    - plot_data: Datenstrukturen für spätere Plots
    """

    # ---------------------------------------------------
    # 1. Feature-Typen bestimmen
    # ---------------------------------------------------
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    onehot_cols = [col for col in X_train.columns if "_" in col and col not in numeric_cols]
    other_cols = [col for col in X_train.columns if col not in numeric_cols + onehot_cols]

    feature_summary = {
        "total_features": len(X_train.columns),
        "numeric_features": len(numeric_cols),
        "onehot_features": len(onehot_cols),
        "other_features": len(other_cols)
    }

    # ---------------------------------------------------
    # 2. Plot-Daten vorbereiten
    # ---------------------------------------------------
    plot_data = {
        "feature_type_counts": {
            "Numerisch": len(numeric_cols),
            "One-Hot": len(onehot_cols),
            "Sonstige": len(other_cols)
        },
        "top10_gini": None,
        "gini_vs_perm": None,
        "correlation_matrix": X_train.corr()
    }

    # ---------------------------------------------------
    # 3. Top-10 Gini Importance
    # ---------------------------------------------------
    if gini_importances is not None:
        plot_data["top10_gini"] = gini_importances.sort_values(
            "gini_importance", ascending=False
        ).head(10)

    # ---------------------------------------------------
    # 4. Vergleich: Gini vs. Permutation Importance
    # ---------------------------------------------------
    if gini_importances is not None and perm_importances is not None:
        merged = gini_importances.merge(perm_importances, on="feature", how="left")
        plot_data["gini_vs_perm"] = merged

    return feature_summary, plot_data
