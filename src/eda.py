import seaborn as sns
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------
# Projektroot ermitteln (eine Ebene über diesem Skript)
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Neuer, sauberer Ordner NUR für EDA-Plots
PLOT_DIR = os.path.join(BASE_DIR, "plots/eda")


def save_plot(name):
    """
    Speichert den aktuellen Plot im plots/eda/-Ordner.
    """
    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, f"{name}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Plot gespeichert: {path}")


def run_eda(df):
    """
    Exploratory Data Analysis (EDA) – bereinigte, portfolio-optimierte Version.
    """

    print("Starte Exploratory Data Analysis...")

    # ---------------------------------------------------
    # 0. Kompakte Info statt HEAD-Ausgabe
    # ---------------------------------------------------
    print(f"EDA gestartet – Datensatzgröße: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    # ---------------------------------------------------
    # 1. Basisinformationen
    # ---------------------------------------------------
    print("\n--- Grundlegende Informationen ---")
    df.info()

    print("\n--- Statistische Übersicht ---")
    print(df.describe())

    print("\n--- Churn-Verteilung (left_company) ---")
    print(df["left_company"].value_counts(normalize=True))

    # ---------------------------------------------------
    # 2. Korrelationen numerischer Variablen
    # ---------------------------------------------------
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    plt.figure(figsize=(12, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=False, cmap="coolwarm")
    plt.title("Correlation Matrix")
    save_plot("correlation_matrix")
    plt.close()

    # ---------------------------------------------------
    # 3. Verteilung relevanter Features
    # ---------------------------------------------------

    # Tenure
    if "years_at_company" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(df["years_at_company"], kde=True)
        plt.title("Distribution of years_at_company")
        save_plot("distribution_years_at_company")
        plt.close()

    # Performance
    if "performance_mean" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(df["performance_mean"], kde=True)
        plt.title("Distribution of performance_mean")
        save_plot("distribution_performance_mean")
        plt.close()

    # Sick Days
    if "sick_days" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(df["sick_days"], kde=False)
        plt.title("Distribution of sick_days")
        save_plot("distribution_sick_days")
        plt.close()

    # Vacation Days
    if "vacation_days" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(df["vacation_days"], kde=False)
        plt.title("Distribution of vacation_days")
        save_plot("distribution_vacation_days")
        plt.close()

    # ---------------------------------------------------
    # 4. Churn-Rate nach Kategorien
    # ---------------------------------------------------
    for col in ["department", "job_level"]:
        if col in df.columns:
            plt.figure(figsize=(6, 4))
            sns.barplot(x=col, y="left_company", data=df)
            plt.title(f"Churn Rate by {col}")
            plt.xticks(rotation=45)
            save_plot(f"churn_by_{col}")
            plt.close()

    print("EDA abgeschlossen.")
