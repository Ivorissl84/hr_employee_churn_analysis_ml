import sys
import os
import pandas as pd
pd.set_option("display.max_columns", None)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)

from src.aggregation import aggregate_data
from src.eda import run_eda
from src.feature_engineering import prepare_features
from src.feature_engineering_visualization import visualize_feature_engineering
from src.model_training import train_model
from src.churn_score import score_employees

import matplotlib.pyplot as plt
import seaborn as sns


def main():
    print("\n=== HR Churn Prediction Pipeline gestartet ===\n")

    # ---------------------------------------------------
    # 1. Daten laden
    # ---------------------------------------------------
    data_path = os.path.join(BASE_DIR, "data", "raw", "hr_data.csv")
    df = pd.read_csv(data_path)
    print(f"Datensatz geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten\n")

    # ---------------------------------------------------
    # 2. Aggregation
    # ---------------------------------------------------
    df_agg, kpis, agg_plots = aggregate_data(df)

    # ---------------------------------------------------
    # 3. Feature Engineering
    # ---------------------------------------------------
    X_train, X_test, y_train, y_test, employee_ids_train, employee_ids_test = prepare_features(df_agg)

    # ---------------------------------------------------
    # 4. Modelltraining
    # ---------------------------------------------------
    model_results = train_model(X_train, y_train, X_test, y_test)

    print("=== MODELLQUALITÄT ===")
    print(f"Accuracy:  {model_results['metrics']['accuracy']:.4f}")
    print(f"Precision: {model_results['metrics']['precision']:.4f}")
    print(f"Recall:    {model_results['metrics']['recall']:.4f}")
    print(f"F1-Score:  {model_results['metrics']['f1']:.4f}\n")

    print("=== CONFUSION MATRIX ===")
    print(model_results["confusion_matrix"], "\n")

    print("=== TOP FEATURES (GINI) ===")
    print(model_results["gini_importance"].head(10), "\n")

    # ---------------------------------------------------
    # 5. Feature Engineering Visualisierung (nur Daten)
    # ---------------------------------------------------
    fe_summary, fe_plots = visualize_feature_engineering(
        X_train,
        model_results["gini_importance"],
        model_results["permutation_importance"]
    )

    print("=== FEATURE ENGINEERING ===")
    print(f"Gesamtzahl Features: {fe_summary['total_features']}")
    print(f"Numerische Features: {fe_summary['numeric_features']}")
    print(f"One-Hot-Features:   {fe_summary['onehot_features']}")
    print(f"Sonstige Features:  {fe_summary['other_features']}\n")

    # ---------------------------------------------------
    # 6. EDA
    # ---------------------------------------------------
    eda_info, eda_plots = run_eda(df_agg)

    print("=== EDA ===")
    print(f"Datensatzgröße: {eda_info['shape']}")
    print("\n--- Churn-Verteilung ---")
    print(eda_info["churn_distribution"], "\n")

    # ---------------------------------------------------
    # 7. HR-KPIs
    # ---------------------------------------------------
    print("=== HR KPIs ===")
    print(f"Churn Rate: {kpis['churn_rate']:.3f}")
    print("\nChurn nach Abteilung:")
    print(kpis["churn_by_department"])
    print("\nChurn nach Job Level:")
    print(kpis["churn_by_job_level"])
    print(f"\nDurchschnittliche Tenure: {kpis['avg_tenure']:.2f}")
    print(f"Durchschnittliche Performance: {kpis['avg_performance']:.2f}")
    print(f"Durchschnittliche Krankheitstage: {kpis['avg_sick_days']:.2f}\n")

    # ---------------------------------------------------
    # 8. Business Impact
    # ---------------------------------------------------
    print("=== BUSINESS IMPACT ===")
    print("Business-Impact-Visualisierung wird am Ende erzeugt.\n")

    # ---------------------------------------------------
    # 9. Churn Scoring
    # ---------------------------------------------------
    df_scored, top_risk, risk_stats = score_employees(
        X_test,
        employee_ids_test,
        model_path=model_results["model_path"]
    )

    print("=== CHURN SCORING ===")
    print(f"Anzahl predicted_churn = 1: {risk_stats['predicted_churn_count']}")
    print("\nRisikoverteilung:")
    print(risk_stats["risk_category_counts"], "\n")

    # ---------------------------------------------------
    # 10. Top-Risiko-Mitarbeiter
    # ---------------------------------------------------
    print("=== TOP 10 GEFÄHRDETE MITARBEITER ===")
    print(top_risk.to_string(index=False))

    # ---------------------------------------------------
    # 11. GANZ AM ENDE: ALLE PLOTS ERZEUGEN
    # ---------------------------------------------------
    print("\nErzeuge Plots...")

    sns.set_theme(style="whitegrid")

    # -------------------------
    # Modellplots
    # -------------------------
    model_plot_dir = os.path.join(BASE_DIR, "plots/model")
    os.makedirs(model_plot_dir, exist_ok=True)

    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(model_results["confusion_matrix"], annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(model_plot_dir, "confusion_matrix.png"))
    plt.close()

    # ROC Curve
    plt.figure(figsize=(6, 5))
    plt.plot(model_results["roc"]["fpr"], model_results["roc"]["tpr"], color="blue")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(os.path.join(model_plot_dir, "roc_curve.png"))
    plt.close()

    # Precision-Recall Curve
    plt.figure(figsize=(6, 5))
    plt.plot(model_results["pr"]["recall"], model_results["pr"]["precision"], color="green")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(os.path.join(model_plot_dir, "precision_recall_curve.png"))
    plt.close()

    # -------------------------
    # Feature Engineering Plots
    # -------------------------
    fe_dir = os.path.join(BASE_DIR, "plots/feature_engineering")
    os.makedirs(fe_dir, exist_ok=True)

    # Feature Type Counts
    plt.figure(figsize=(6, 4))
    sns.barplot(
        x=list(fe_plots["feature_type_counts"].keys()),
        y=list(fe_plots["feature_type_counts"].values()),
        color="steelblue"
    )
    plt.title("Feature Types After Engineering")
    plt.ylabel("Count")
    plt.savefig(os.path.join(fe_dir, "feature_type_counts.png"))
    plt.close()

    # Top-10 Feature Importances
    top10 = model_results["gini_importance"].head(10)
    plt.figure(figsize=(8, 5))
    sns.barplot(
        x="gini_importance",
        y="feature",
        data=top10,
        color="darkgreen"
    )
    plt.title("Top 10 Feature Importances (Gini)")
    plt.xlabel("Gini Importance")
    plt.ylabel("Feature")
    plt.savefig(os.path.join(fe_dir, "feature_importance_top10.png"))
    plt.close()

    # Gini vs Permutation Importance
    merged = model_results["gini_importance"].merge(
        model_results["permutation_importance"], on="feature", how="left"
    )
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x="gini_importance",
        y="perm_importance",
        data=merged,
        color="darkorange"
    )
    plt.title("Gini vs. Permutation Importance")
    plt.xlabel("Gini Importance")
    plt.ylabel("Permutation Importance")
    plt.savefig(os.path.join(fe_dir, "feature_importance_stability.png"))
    plt.close()

    # -------------------------
    # Business Impact Plots
    # -------------------------
    bi_dir = os.path.join(BASE_DIR, "plots/aggregation")
    os.makedirs(bi_dir, exist_ok=True)

    # Heatmap Department × Job Level
    plt.figure(figsize=(10, 6))
    sns.heatmap(agg_plots["pivot_churn_dept_job"], annot=True, cmap="Reds", fmt=".2f")
    plt.title("Churn Rate by Department × Job Level")
    plt.savefig(os.path.join(bi_dir, "churn_department_joblevel.png"))
    plt.close()

    # Churn vs Tenure
    if "years_at_company" in df_agg.columns:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(
            x=df_agg["years_at_company"],
            y=df_agg["left_company"],
            color="purple",
            alpha=0.6
        )
        plt.title("Churn vs. Tenure")
        plt.xlabel("Years at Company")
        plt.ylabel("Churn (0/1)")
        plt.savefig(os.path.join(bi_dir, "churn_vs_tenure.png"))
        plt.close()

    print("\n=== Pipeline erfolgreich abgeschlossen ===")


if __name__ == "__main__":
    main()
