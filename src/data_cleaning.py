import pandas as pd

def clean_data(employees, absences, reviews, events):
    """
    Bereinigt und aggregiert die Rohdaten aus Employees, Absences,
    Performance Reviews und HR Events.
    Schritte:
    - Grundbereinigung (Duplikate, Datumsformate)
    - Aggregation von Abwesenheiten
    - Aggregation von Performance-Daten
    - Aggregation von HR-Events
    - Zusammenführen aller Teil-DataFrames
    """

    print("Starte Datenbereinigung und Aggregation...")

    # ---------------------------------------------------
    # 1. Grundbereinigung
    # ---------------------------------------------------
    # Datumsfelder in echtes Datumsformat konvertieren
    absences["date"] = pd.to_datetime(absences["date"], errors="coerce")
    events["date"] = pd.to_datetime(events["date"], errors="coerce")
    reviews["date"] = pd.to_datetime(reviews["date"], errors="coerce")

    # Doppelte Einträge entfernen
    employees = employees.drop_duplicates(subset="employee_id")
    absences = absences.drop_duplicates()
    reviews = reviews.drop_duplicates()
    events = events.drop_duplicates()

    # ---------------------------------------------------
    # 2. Abwesenheiten aggregieren
    # ---------------------------------------------------
    # Anzahl Abwesenheiten pro Typ (z. B. sick, vacation, accident)
    abs_type_counts = absences.pivot_table(
        index="employee_id",
        columns="absence_type",
        values="duration_days",
        aggfunc="sum",
        fill_value=0
    ).rename(columns={
        "sick": "sick_days",
        "vacation": "vacation_days",
        "accident": "accident_days",
        "other": "other_absence_days"
    })

    # Grundlegende Abwesenheitskennzahlen
    abs_basic = absences.groupby("employee_id").agg(
        total_absences=("duration_days", "count"),
        total_absence_days=("duration_days", "sum"),
        avg_absence_duration=("duration_days", "mean")
    )

    # Alle Abwesenheitsdaten zusammenführen
    abs_agg = abs_basic.join(abs_type_counts, how="left").fillna(0)

    # ---------------------------------------------------
    # 3. Performance aggregieren
    # ---------------------------------------------------
    # Durchschnitt, Minimum, Maximum und Bonus
    perf_agg = reviews.groupby("employee_id").agg(
        performance_avg=("performance_score", "mean"),
        performance_min=("performance_score", "min"),
        performance_max=("performance_score", "max"),
        bonus_avg=("bonus_percent", "mean")
    )

    # Performance-Trend: letzte Bewertung minus Durchschnitt
    last_review = (
        reviews.sort_values(["employee_id", "date"])
        .groupby("employee_id")
        .tail(1)
        .set_index("employee_id")
    )

    perf_agg["performance_trend"] = (
        last_review["performance_score"] - perf_agg["performance_avg"]
    )

    # ---------------------------------------------------
    # 4. HR-Events aggregieren
    # ---------------------------------------------------
    # Anzahl Events pro Typ (z. B. promotion, warning, training)
    events_agg = (
        events.groupby(["employee_id", "event_type"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={
            "promotion": "promotions",
            "warning": "warnings",
            "training": "trainings",
            "team_change": "team_changes"
        })
    )

    # ---------------------------------------------------
    # 5. Alle Teil-DataFrames zusammenführen
    # ---------------------------------------------------
    df = employees.copy()

    df = df.merge(abs_agg, on="employee_id", how="left")
    df = df.merge(perf_agg, on="employee_id", how="left")
    df = df.merge(events_agg, on="employee_id", how="left")

    # Fehlende Werte auffüllen
    df = df.fillna(0)

    print("Datenbereinigung abgeschlossen.")
    return df
