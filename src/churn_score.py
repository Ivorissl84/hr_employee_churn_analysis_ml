import pandas as pd
import joblib


def load_model(model_path="model/random_forest_model.pkl"):
    """
    Lädt das trainierte Random-Forest-Modell aus einer .pkl-Datei.
    """
    print(f"Lade Modell aus: {model_path}")
    model = joblib.load(model_path)
    return model


def predict_churn(df, model):
    """
    Berechnet Churn-Wahrscheinlichkeiten für alle Mitarbeiter im DataFrame.
    Erwartet:
        DataFrame mit denselben Features wie im Training.
    Gibt:
        Array mit Wahrscheinlichkeiten für Klasse 1 (Kündigung).
    """
    print("Berechne Churn-Wahrscheinlichkeiten...")
    churn_scores = model.predict_proba(df)[:, 1]
    return churn_scores


def add_risk_categories(df, scores, threshold=0.35):
    """
    Fügt dem DataFrame folgende Informationen hinzu:
    - churn_score: vorhergesagte Kündigungswahrscheinlichkeit
    - predicted_churn: binäre Vorhersage basierend auf Schwelle
    - risk_category: Einteilung in low / medium / high Risiko
    """

    df = df.copy()
    df["churn_score"] = scores

    # Binäre Vorhersage basierend auf einstellbarer Schwelle
    df["predicted_churn"] = (df["churn_score"] >= threshold).astype(int)

    # Risiko-Kategorien definieren
    df["risk_category"] = pd.cut(
        df["churn_score"],
        bins=[0.0, 0.25, 0.40, 1.0],
        labels=["low", "medium", "high"],
        include_lowest=True
    )

    return df


def get_top_risk(df, n=10):
    """
    Gibt die Top-N Mitarbeiter mit dem höchsten Kündigungsrisiko zurück.
    Sortiert nach churn_score absteigend.
    """
    return df.sort_values("churn_score", ascending=False).head(n)


def score_employees(df, model_path="model/random_forest_model.pkl", top_n=10, threshold=0.35):
    """
    Hauptfunktion für das Churn-Scoring:
    - Modell laden
    - Churn-Wahrscheinlichkeiten berechnen
    - Risiko-Kategorien hinzufügen
    - Analyse der vorhergesagten Kündigungen
    - Top-Risiko-Mitarbeiter bestimmen
    """

    print("Starte Churn-Scoring...")

    # ---------------------------------------------------
    # 1. Modell laden
    # ---------------------------------------------------
    model = load_model(model_path)

    # ---------------------------------------------------
    # 2. Churn-Scores berechnen
    # ---------------------------------------------------
    scores = predict_churn(df, model)

    # ---------------------------------------------------
    # 3. Risiko-Kategorien hinzufügen
    # ---------------------------------------------------
    df_scored = add_risk_categories(df, scores, threshold=threshold)

    # ---------------------------------------------------
    # 4. Analyse der vorhergesagten Kündigungen
    # ---------------------------------------------------
    predicted_churn_count = df_scored["predicted_churn"].sum()
    print(f"\n=== Analyse: Vorhergesagte Kündigungen ===")
    print(f"Anzahl predicted_churn = 1: {predicted_churn_count}")

    print("\n=== Risiko-Verteilung ===")
    print(df_scored["risk_category"].value_counts())

    # ---------------------------------------------------
    # 5. Top-Risiko-Mitarbeiter bestimmen
    # ---------------------------------------------------
    top_risk = get_top_risk(df_scored, n=top_n)

    print("\nChurn-Scoring abgeschlossen.")
    return df_scored, top_risk
