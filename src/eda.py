import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO


def run_eda(df):
    """
    Führt eine EDA durch, OHNE etwas zu drucken oder zu plotten.
    Gibt zurück:
    - eda_info: Basisinformationen & Statistiken
    - plot_data: Datenstrukturen für spätere Plots
    """

    # ---------------------------------------------------
    # 1. Basisinformationen sammeln (stabiler Info‑Fix)
    # ---------------------------------------------------
    buffer = StringIO()
    df.info(buf=buffer)
    info_text = buffer.getvalue()

    eda_info = {
        "shape": df.shape,
        "info": info_text,
        "describe": df.describe(include="all"),
        "churn_distribution": df["left_company"].value_counts(normalize=True)
    }

    # ---------------------------------------------------
    # 2. Plot-Daten vorbereiten (keine Plots!)
    # ---------------------------------------------------
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    plot_data = {
        "correlation_matrix": df[numeric_cols].corr(),
        "distribution_years_at_company": df["years_at_company"] if "years_at_company" in df.columns else None,
        "distribution_performance_mean": df["performance_mean"] if "performance_mean" in df.columns else None,
        "distribution_sick_days": df["sick_days"] if "sick_days" in df.columns else None,
        "distribution_vacation_days": df["vacation_days"] if "vacation_days" in df.columns else None,
        "churn_by_department": df.groupby("department")["left_company"].mean() if "department" in df.columns else None,
        "churn_by_job_level": df.groupby("job_level")["left_company"].mean() if "job_level" in df.columns else None
    }

    return eda_info, plot_data
