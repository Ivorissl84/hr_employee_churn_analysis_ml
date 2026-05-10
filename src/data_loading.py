import pandas as pd
import os

DATA_PATH = "data"


def load_csv(filename):
    """
    Lädt eine einzelne CSV-Datei aus dem data/-Ordner.
    Führt einen einfachen Existenz-Check durch und gibt den DataFrame zurück.
    """
    path = os.path.join(DATA_PATH, filename)

    # Sicherstellen, dass die Datei existiert
    if not os.path.exists(path):
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")

    # CSV laden
    df = pd.read_csv(path)
    print(f"{filename} geladen – {df.shape[0]} Zeilen, {df.shape[1]} Spalten")
    return df


def load_all_data():
    """
    Lädt alle relevanten HR-Datensätze:
    - employees.csv
    - absences.csv
    - performance_reviews.csv
    - hr_events.csv

    Gibt vier DataFrames zurück.
    """

    print("Lade alle HR-Daten...")

    # ---------------------------------------------------
    # 1. Einzelne CSV-Dateien laden
    # ---------------------------------------------------
    employees = load_csv("employees.csv")
    absences = load_csv("absences.csv")
    reviews = load_csv("performance_reviews.csv")
    events = load_csv("hr_events.csv")

    # ---------------------------------------------------
    # 2. Rückgabe aller geladenen DataFrames
    # ---------------------------------------------------
    print("Alle Daten erfolgreich geladen.")
    return employees, absences, reviews, events
