import pandas as pd
import numpy as np

np.random.seed(42)

# Mitarbeiterdaten laden
employees = pd.read_csv("../data/employees.csv")

# Zeitraum definieren
start_date = pd.Timestamp("2021-01-01")
end_date = pd.Timestamp("2024-12-31")
days_range = (end_date - start_date).days

rows = []

def random_date():
    return start_date + pd.Timedelta(days=np.random.randint(0, days_range))

for _, row in employees.iterrows():
    emp_id = row["employee_id"]
    perf = row["performance_score"]
    job_sat = row["job_satisfaction"]
    mgmt_sat = row["management_satisfaction"]
    trainings = row["trainings_per_year"]

    # Anzahl Events pro Mitarbeiter (2–5)
    n_events = np.random.randint(2, 6)

    for _ in range(n_events):

        # Wahrscheinlichkeiten abhängig von Mitarbeiterdaten
        probs = {
            "promotion": 0.02 + 0.05 * (perf - 3),  # Score 5 → 12%, Score 1 → 0%
            "warning": 0.02 + 0.04 * (3 - min(job_sat, mgmt_sat)),  # schlechte Zufriedenheit → mehr Warnings
            "training": 0.40 + 0.05 * trainings,  # viele Trainings → mehr Events
            "team_change": 0.10
        }

        # Negative Werte verhindern
        for k in probs:
            probs[k] = max(0, probs[k])

        # Normalisieren
        total = sum(probs.values())
        for k in probs:
            probs[k] /= total

        event_type = np.random.choice(list(probs.keys()), p=list(probs.values()))

        rows.append({
            "employee_id": emp_id,
            "date": random_date(),
            "event_type": event_type
        })

df = pd.DataFrame(rows)

# Datei speichern
df.to_csv("../data/hr_events.csv", index=False)

print("hr_events.csv wurde erfolgreich erzeugt!")
