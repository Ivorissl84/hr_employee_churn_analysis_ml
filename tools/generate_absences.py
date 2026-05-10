import pandas as pd
import numpy as np

np.random.seed(42)

# Anzahl Mitarbeiter (muss zu employees.csv passen)
n_employees = 300

# Anzahl Abwesenheitseinträge
n_absences = 1200  # ca. 4 pro Mitarbeiter im Schnitt

absence_types = ["sick", "vacation", "accident", "other"]

# Zeitraum definieren
start_date = pd.Timestamp("2022-01-01")
end_date = pd.Timestamp("2024-12-31")
days_range = (end_date - start_date).days

# Zufällige Abwesenheiten erzeugen
data = {
    "employee_id": np.random.randint(1, n_employees + 1, size=n_absences),
    "date": [start_date + pd.Timedelta(days=int(x)) for x in np.random.randint(0, days_range, size=n_absences)],
    "absence_type": np.random.choice(absence_types, size=n_absences, p=[0.55, 0.35, 0.05, 0.05]),
    "duration_days": np.random.randint(1, 11, size=n_absences)
}

df = pd.DataFrame(data)

# Datei speichern
df.to_csv("../data/absences.csv", index=False)

print("absences.csv wurde erfolgreich erzeugt!")
