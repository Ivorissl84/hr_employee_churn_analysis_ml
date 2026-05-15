import pandas as pd
import joblib


def load_model(model_path="model/random_forest_model.pkl"):
    """Lädt das Modell ohne Konsolenausgabe."""
    return joblib.load(model_path)


def predict_churn(df, model):
    """Berechnet Churn-Wahrscheinlichkeiten ohne Prints."""
    return model.predict_proba(df)[:, 1]


def add_risk_categories(df, scores, threshold=0.35):
    """Fügt Risiko-Kategorien hinzu – ohne Prints."""
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
    """Gibt die Top-N Risiko-Mitarbeiter zurück."""
    return df.sort_values("churn_score", ascending=False).head(n)


def score_employees(df, employee_ids, model_path="model/random_forest_model.pkl", top_n=10, threshold=0.35):
    """
    Führt das Churn-Scoring durch – ohne Prints.
    Gibt zurück:
    - df_scored: vollständiger Score-Datensatz
    - top_risk: Top-N Risiko-Mitarbeiter
    - risk_stats: Kennzahlen zur Risiko-Verteilung
    """

    # 1. Modell laden
    model = load_model(model_path)

    # 2. Churn-Scores berechnen
    scores = predict_churn(df, model)

    # 3. Risiko-Kategorien hinzufügen
    df_scored = add_risk_categories(df, scores, threshold=threshold)

    # employee_id wieder anhängen
    df_scored["employee_id"] = employee_ids.values

    # 4. Risiko-Statistiken
    risk_stats = {
        "predicted_churn_count": int(df_scored["predicted_churn"].sum()),
        "risk_category_counts": df_scored["risk_category"].value_counts()
    }

    # 5. Top-Risiko-Mitarbeiter bestimmen
    top_risk = get_top_risk(df_scored, n=top_n)

    # 6. Department-Dummys zurückwandeln
    department_cols = [c for c in top_risk.columns if c.startswith("department_")]

    if department_cols:
        top_risk["department"] = (
            top_risk[department_cols]
            .idxmax(axis=1)
            .str.replace("department_", "", regex=False)
        )
        top_risk = top_risk.drop(columns=department_cols)

    # employee_id nach vorne holen
    cols = ["employee_id"] + [c for c in top_risk.columns if c != "employee_id"]
    top_risk = top_risk[cols]

    return df_scored, top_risk, risk_stats
