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


def main():
    print("\n=== HR Churn Prediction Pipeline gestartet ===\n")

    # 1. Daten laden
    print("Lade Event-Daten...")
    data_path = os.path.join(BASE_DIR, "data", "raw", "hr_data.csv")
    df = pd.read_csv(data_path)
    print(f"Datensatz geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten\n")

    # 2. Aggregation (HR-KPIs & Business Impact)
    print("Starte Aggregation...")
    df_agg = aggregate_data(df)
    print("Aggregation abgeschlossen.\n")

    # 3. Feature Engineering
    print("Starte Feature Engineering...")
    X_train, X_test, y_train, y_test, employee_ids_train, employee_ids_test = prepare_features(df_agg)
    print("Feature Engineering abgeschlossen.\n")

    # 4. Modelltraining
    print("=== MODELLQUALITÄT ===")
    model, feature_importances = train_model(X_train, y_train, X_test, y_test)
    print("Modelltraining abgeschlossen.\n")

    # 5. Feature Engineering Visualisierung
    print("=== FEATURE ENGINEERING ===")
    visualize_feature_engineering(X_train, feature_importances)
    print("Feature Engineering Visualisierung abgeschlossen.\n")

    # 6. EDA
    print("=== EDA ===")
    run_eda(df_agg)
    print("EDA abgeschlossen.\n")

    # 7. HR-KPIs (aus Aggregation)
    print("=== HR KPIs ===")
    print("HR-Kennzahlen wurden oben ausgegeben.\n")

    # 8. Business Impact
    print("=== BUSINESS IMPACT ===")
    print("Business-Impact-Plots wurden gespeichert.\n")

    # 9. Churn Scoring
    print("=== CHURN SCORING ===")
    df_scored, top_risk = score_employees(X_test, employee_ids_test)
    print("Churn Scoring abgeschlossen.\n")

    # 10. Top-Risiko-Mitarbeiter
    print("=== TOP 10 GEFÄHRDETE MITARBEITER ===")
    print(top_risk.to_string(index=False))

    print("\n=== Pipeline erfolgreich abgeschlossen ===")


if __name__ == "__main__":
    main()
