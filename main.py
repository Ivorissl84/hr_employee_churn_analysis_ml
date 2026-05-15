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
    # 2. Aggregation (liefert df_agg, KPIs, Plotdaten)
    # ---------------------------------------------------
    df_agg, kpis, agg_plots = aggregate_data(df)

    # ---------------------------------------------------
    # 3. Feature Engineering
    # ---------------------------------------------------
    X_train, X_test, y_train, y_test, employee_ids_train, employee_ids_test = prepare_features(df_agg)

    # ---------------------------------------------------
    # 4. Modelltraining (Modellqualität zuerst!)
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
    # 6. EDA (nur Daten)
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
    # 8. Business Impact (Plots kommen später)
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

    # Modellplots
    model_plot_dir = os.path.join(BASE_DIR, "plots/model")
    os.makedirs(model_plot_dir, exist_ok=True)

    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(model_results["confusion_matrix"], annot=True, fmt="d", cmap="Greys")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(model_plot_dir, "confusion_matrix.png"))
    plt.close()

    # ROC
    plt.figure(figsize=(6, 5))
    plt.plot(model_results["roc"]["fpr"], model_results["roc"]["tpr"])
    plt.title("ROC Curve")
    plt.savefig(os.path.join(model_plot_dir, "roc_curve.png"))
    plt.close()

    # Precision-Recall
    plt.figure(figsize=(6, 5))
    plt.plot(model_results["pr"]["recall"], model_results["pr"]["precision"])
    plt.title("Precision-Recall Curve")
    plt.savefig(os.path.join(model_plot_dir, "precision_recall_curve.png"))
    plt.close()

    # Feature Engineering Plots
    fe_dir = os.path.join(BASE_DIR, "plots/feature_engineering")
    os.makedirs(fe_dir, exist_ok=True)

    # Feature Type Counts
    plt.figure(figsize=(6, 4))
    sns.barplot(
        x=list(fe_plots["feature_type_counts"].keys()),
        y=list(fe_plots["feature_type_counts"].values())
    )
    plt.title("Feature-Typen")
    plt.savefig(os.path.join(fe_dir, "feature_type_counts.png"))
    plt.close()

    # Business Impact Plots
    bi_dir = os.path.join(BASE_DIR, "plots/aggregation")
    os.makedirs(bi_dir, exist_ok=True)

    # Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(agg_plots["pivot_churn_dept_job"], annot=True, cmap="Reds")
    plt.title("Churn nach Department × Job Level")
    plt.savefig(os.path.join(bi_dir, "churn_department_joblevel.png"))
    plt.close()

    print("\n=== Pipeline erfolgreich abgeschlossen ===")


if __name__ == "__main__":
    main()
