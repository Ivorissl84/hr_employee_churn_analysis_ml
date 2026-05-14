import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


def aggregate_data(df):
    """
    Aggregiert Event-basierte HR-Daten auf Mitarbeiterebene.
    Erwartet: DataFrame mit Spalten wie employee_id, event_type, value, date, etc.
    Gibt: Einen DataFrame mit einem Eintrag pro Mitarbeiter zurück.
    """

    print("Aggregiere Daten auf Mitarbeiterebene...")

    # ---------------------------------------------------
    # Datum sicherstellen
    # ---------------------------------------------------
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # ---------------------------------------------------
    # Basis: Ein Eintrag pro Mitarbeiter
    # ---------------------------------------------------
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

    # ---------------------------------------------------
    # Abwesenheiten
    # ---------------------------------------------------
    absences = df[df["event_type"] == "absence"]

    sick_days = absences[absences["absence_type"] == "sick"] \
        .groupby("employee_id").size().rename("sick_days")

    vacation_days = absences[absences["absence_type"] == "vacation"] \
        .groupby("employee_id").size().rename("vacation_days")

    # ---------------------------------------------------
    # HR Events
    # ---------------------------------------------------
    warnings = df[df["event_type"] == "warning"].groupby("employee_id").size().rename("warnings")
    trainings = df[df["event_type"] == "training"].groupby("employee_id").size().rename("trainings")
    team_changes = df[df["event_type"] == "team_change"].groupby("employee_id").size().rename("team_changes")

    # ---------------------------------------------------
    # Performance
    # ---------------------------------------------------
    performance_mean = df[df["event_type"] == "performance"] \
        .groupby("employee_id")["value"].mean().rename("performance_mean")

    performance_trend = df[df["event_type"] == "performance"] \
        .groupby("employee_id")["value"].agg(
            lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0
        ).rename("performance_trend")

    # ---------------------------------------------------
    # Alles zusammenführen
    # ---------------------------------------------------
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

    # ---------------------------------------------------
    # HR-KPIs (kompakt, ohne HEAD)
    # ---------------------------------------------------
    print("\n=== HR KPIs ===")

    churn_rate = aggregated["left_company"].mean()
    print(f"Churn Rate: {churn_rate:.3f}")

    churn_by_department = aggregated.groupby("department")["left_company"].mean().sort_values(ascending=False)
    print("\nChurn nach Abteilung:")
    print(churn_by_department)

    churn_by_job_level = aggregated.groupby("job_level")["left_company"].mean().sort_values(ascending=False)
    print("\nChurn nach Job Level:")
    print(churn_by_job_level)

    avg_tenure = aggregated["years_at_company"].mean()
    print(f"\nDurchschnittliche Tenure (Jahre): {avg_tenure:.2f}")

    avg_performance = aggregated["performance_mean"].mean()
    print(f"Durchschnittliche Performance: {avg_performance:.2f}")

    avg_sick_days = aggregated["sick_days"].mean()
    print(f"Durchschnittliche Krankheitstage: {avg_sick_days:.2f}")

    print("\nAggregation abgeschlossen.")

    # ---------------------------------------------------
    # BUSINESS-IMPACT-PLOTS
    # ---------------------------------------------------
    plot_dir = "plots/aggregation"
    os.makedirs(plot_dir, exist_ok=True)

    # 1. Heatmap: Churn nach Department × Job Level
    pivot = aggregated.pivot_table(
        values="left_company",
        index="department",
        columns="job_level",
        aggfunc="mean"
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, cmap="Reds", fmt=".2f")
    plt.title("Churn-Rate nach Department und Job Level")
    heatmap_path = os.path.join(plot_dir, "churn_department_joblevel.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot gespeichert: {heatmap_path}")

    # 2. Scatterplot: Churn vs. Tenure
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=aggregated,
        x="years_at_company",
        y="left_company",
        hue="left_company",
        palette=["grey", "red"]
    )
    plt.title("Churn vs. Tenure")
    plt.xlabel("Years at Company")
    plt.ylabel("Churn (0/1)")
    tenure_path = os.path.join(plot_dir, "churn_vs_tenure.png")
    plt.savefig(tenure_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot gespeichert: {tenure_path}")

    # 3. Boxplot: Churn vs. Performance Trend (korrigiert)
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=aggregated,
        x="left_company",
        y="performance_trend",
        hue="left_company",
        palette=["grey", "red"],
        legend=False
    )
    plt.title("Churn vs. Performance Trend")
    plt.xlabel("Churn (0/1)")
    plt.ylabel("Performance Trend")
    perf_path = os.path.join(plot_dir, "churn_vs_performance_trend.png")
    plt.savefig(perf_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot gespeichert: {perf_path}")

    return aggregated
