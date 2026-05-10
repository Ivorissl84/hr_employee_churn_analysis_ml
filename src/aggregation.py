import pandas as pd
import numpy as np


def aggregate_data(df):
    """
    Aggregiert Event-basierte HR-Daten auf Mitarbeiterebene.
    Erwartet: DataFrame mit Spalten wie employee_id, event_type, value, date, etc.
    Gibt: Einen DataFrame mit einem Eintrag pro Mitarbeiter zurück.
    """

    print("Aggregiere Daten auf Mitarbeiterebene...")

    # Sicherstellen, dass Datum ein echtes Datum ist
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Basis: Ein Eintrag pro Mitarbeiter
    base = df.groupby("employee_id").agg({
        "age": "first",
        "gender": "first",
        "department": "first",
        "job_level": "first",
        "salary": "first",
        "hire_date": "first",
        "left_company": "first"
    }).reset_index()

    # Dienstjahre berechnen
    if "hire_date" in df.columns:
        base["hire_date"] = pd.to_datetime(base["hire_date"], errors="coerce")
        base["years_at_company"] = (pd.Timestamp.today() - base["hire_date"]).dt.days / 365.25

    # -----------------------------
    # Abwesenheiten (Absences)
    # -----------------------------
    absences = df[df["event_type"] == "absence"]

    sick_days = absences[absences["absence_type"] == "sick"] \
        .groupby("employee_id").size().rename("sick_days")

    vacation_days = absences[absences["absence_type"] == "vacation"] \
        .groupby("employee_id").size().rename("vacation_days")

    # -----------------------------
    # HR Events
    # -----------------------------
    warnings = df[df["event_type"] == "warning"].groupby("employee_id").size().rename("warnings")
    trainings = df[df["event_type"] == "training"].groupby("employee_id").size().rename("trainings")
    team_changes = df[df["event_type"] == "team_change"].groupby("employee_id").size().rename("team_changes")

    # -----------------------------
    # Performance
    # -----------------------------
    performance_mean = df[df["event_type"] == "performance"] \
        .groupby("employee_id")["value"].mean().rename("performance_mean")

    performance_trend = df[df["event_type"] == "performance"] \
        .groupby("employee_id")["value"].agg(
            lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0
        ).rename("performance_trend")

    # -----------------------------
    # Alles zusammenführen
    # -----------------------------
    aggregated = base \
        .merge(sick_days, on="employee_id", how="left") \
        .merge(vacation_days, on="employee_id", how="left") \
        .merge(warnings, on="employee_id", how="left") \
        .merge(trainings, on="employee_id", how="left") \
        .merge(team_changes, on="employee_id", how="left") \
        .merge(performance_mean, on="employee_id", how="left") \
        .merge(performance_trend, on="employee_id", how="left")

    # Fehlende Werte auffüllen
    aggregated = aggregated.fillna({
        "sick_days": 0,
        "vacation_days": 0,
        "warnings": 0,
        "trainings": 0,
        "team_changes": 0,
        "performance_mean": aggregated["performance_mean"].mean(),
        "performance_trend": 0
    })

    print("Aggregation abgeschlossen.")
    return aggregated
