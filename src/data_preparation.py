import pandas as pd
import os


def prepare_raw_dataset():
    """
    Führt employees.csv, absences.csv, hr_events.csv und performance_reviews.csv
    zu einem einheitlichen Event-basierten HR-Dataset zusammen.
    Speichert das Ergebnis als data/raw/hr_data.csv (immer im Projekt-Root).
    """

    print("Starte Datenzusammenführung...")

    # ---------------------------------------------------
    # 1. Basis-Pfade korrekt bestimmen (egal von wo gestartet)
    # ---------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    RAW_DIR = os.path.join(DATA_DIR, "raw")

    os.makedirs(RAW_DIR, exist_ok=True)

    # ---------------------------------------------------
    # 2. Dateien laden
    # ---------------------------------------------------
    employees = pd.read_csv(os.path.join(DATA_DIR, "employees.csv"))
    absences = pd.read_csv(os.path.join(DATA_DIR, "absences.csv"))
    events = pd.read_csv(os.path.join(DATA_DIR, "hr_events.csv"))
    performance = pd.read_csv(os.path.join(DATA_DIR, "performance_reviews.csv"))

    # ---------------------------------------------------
    # 3. Employees vorbereiten
    # ---------------------------------------------------
    employees.columns = employees.columns.str.lower().str.strip()

    # hire_date aus years_at_company berechnen
    employees["hire_date"] = pd.Timestamp.today() - pd.to_timedelta(
        employees["years_at_company"] * 365, unit="D"
    )

    # nur die Spalten behalten, die wir für die Pipeline brauchen
    base_cols = [
        "employee_id", "age", "gender", "department",
        "job_level", "salary", "hire_date", "left_company"
    ]

    employees = employees[base_cols]

    # ---------------------------------------------------
    # 4. Absences → Events umwandeln
    # ---------------------------------------------------
    absences.columns = absences.columns.str.lower().str.strip()

    absences["event_type"] = "absence"
    absences["value"] = absences["duration_days"]

    # WICHTIG: absence_type behalten!
    absences = absences[[
        "employee_id",
        "date",
        "event_type",
        "value",
        "absence_type"
    ]]

    # ---------------------------------------------------
    # 5. HR Events vereinheitlichen
    # ---------------------------------------------------
    events.columns = events.columns.str.lower().str.strip()
    if "value" not in events.columns:
        events["value"] = 1
    events = events[["employee_id", "date", "event_type", "value"]]

    # ---------------------------------------------------
    # 6. Performance Reviews vereinheitlichen
    # ---------------------------------------------------
    performance.columns = performance.columns.str.lower().str.strip()
    performance["event_type"] = "performance"
    performance.rename(columns={"performance_score": "value"}, inplace=True)
    performance = performance[["employee_id", "date", "event_type", "value"]]

    # ---------------------------------------------------
    # 7. Alles zu einem Event-Dataset kombinieren
    # ---------------------------------------------------
    combined = pd.concat([absences, events, performance], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")

    # ---------------------------------------------------
    # 8. Employees mit Events mergen
    # ---------------------------------------------------
    hr_data = combined.merge(employees, on="employee_id", how="left")

    # ---------------------------------------------------
    # 9. Speichern
    # ---------------------------------------------------
    output_path = os.path.join(RAW_DIR, "hr_data.csv")
    hr_data.to_csv(output_path, index=False)

    print(f"hr_data.csv erfolgreich erstellt: {len(hr_data)} Zeilen")
    print(f"Gespeichert unter: {output_path}")

    print("Max age:", hr_data["age"].max())
    print("Correlation age vs salary:", hr_data["age"].corr(hr_data["salary"]))
    print("Correlation age vs left_company:", hr_data["age"].corr(hr_data["left_company"]))

    return hr_data


if __name__ == "__main__":
    prepare_raw_dataset()