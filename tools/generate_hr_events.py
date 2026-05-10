import pandas as pd
import numpy as np

np.random.seed(42)

# Anzahl Mitarbeiter (muss zu employees.csv passen)
n_employees = 300

# Anzahl HR-Events insgesamt
n_events = 900  # ca. 3 Events pro Mitarbeiter

event_types = ["promotion", "warning", "training", "team_change"]

# Zeitraum definieren
start_date = pd.Timestamp("2021-01-01")
end_date = pd.Timestamp("2024-12-31")
days_range = (end_date - start_date).days

# Zufällige Events erzeugen
data = {
    "employee_id": np.random.randint(1, n_employees + 1, size=n_events),
    "date": [start_date + pd.Timedelta(days=int(x)) for x in np.random.randint(0, days_range, size=n_events)],
    "event_type": np.random.choice(event_types, size=n_events, p=[0.20, 0.10, 0.50, 0.20])
}

df = pd.DataFrame(data)

# Datei speichern
df.to_csv("../data/hr_events.csv", index=False)

print("hr_events.csv wurde erfolgreich erzeugt!")
