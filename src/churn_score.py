import pandas as pd
import joblib


def load_model(model_path="model/random_forest_model.pkl"):
    print(f"Lade Modell aus: {model_path}")
    model = joblib.load(model_path)
    return model


def predict_churn(df, model):
    print("Berechne Churn-Wahrscheinlichkeiten...")
    churn_scores = model.predict_proba(df)[:, 1]
    return churn_scores


def add_risk_categories(df, scores, threshold=0.35):
    df = df.copy()
    df["churn_score"] = scores
    df["predicted_churn"] = (df["churn_score"] >= threshold).astype(int)

    df["risk_category"] = pd.cut(
        df["churn_score"],
        bins=[0.0, 0.25, 0.40, 1.0],
        labels=["low", "medium", "high"],
        include_lowest=True
    )

    return df


def get_top_risk(df, n=10):
    return df.sort_values("churn_score", ascending=False).head(n)


def score_employees(df, employee_ids, model_path="model/random_forest_model.pkl", top_n=10, threshold=0.35):
    """
    employee_ids MUSS jetzt mitgegeben werden!
    """

    print("Starte Churn-Scoring...")

    # 1. Modell laden
    model = load_model(model_path)

    # 2. Churn-Scores berechnen
    scores = predict_churn(df, model)

    # 3. Risiko-Kategorien hinzufügen
    df_scored = add_risk_categories(df, scores, threshold=threshold)

    # ---------------------------------------------------
    # >>> NEU <<<
    # employee_id wieder anhängen
    # ---------------------------------------------------
    df_scored["employee_id"] = employee_ids.values

    # 4. Analyse der vorhergesagten Kündigungen
    predicted_churn_count = df_scored["predicted_churn"].sum()
    print("\n=== ANALYSE: VORHERGESAGTE KÜNDIGUNGEN ===")
    print(f"Anzahl predicted_churn = 1: {predicted_churn_count}")

    # 5. Risiko-Verteilung
    print("\n=== RISIKO-VERTEILUNG ===")
    print(df_scored["risk_category"].value_counts())

    # 6. Top-Risiko-Mitarbeiter bestimmen
    top_risk = get_top_risk(df_scored, n=top_n)

    # ---------------------------------------------------
    # 7. Department-Dummys zurück in eine Spalte wandeln
    # ---------------------------------------------------
    department_cols = [c for c in top_risk.columns if c.startswith("department_")]

    if department_cols:
        top_risk["department"] = (
            top_risk[department_cols]
            .idxmax(axis=1)
            .str.replace("department_", "", regex=False)
        )
        top_risk = top_risk.drop(columns=department_cols)

    # ---------------------------------------------------
    # 8. employee_id nach vorne holen
    # ---------------------------------------------------
    cols = ["employee_id"] + [c for c in top_risk.columns if c != "employee_id"]
    top_risk = top_risk[cols]

    print("\nChurn-Scoring abgeschlossen.")
    return df_scored, top_risk
