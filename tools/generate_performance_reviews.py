import pandas as pd
import numpy as np

np.random.seed(42)

# Anzahl Mitarbeiter (muss zu employees.csv passen)
n_employees = 300

# Jahre für Bewertungen
years = [2021, 2022, 2023, 2024]

rows = []

for emp_id in range(1, n_employees + 1):
    # Jeder Mitarbeiter bekommt 2–4 Bewertungen
    n_reviews = np.random.randint(2, 5)
    review_years = np.random.choice(years, size=n_reviews, replace=False)

    for year in review_years:
        performance_score = np.random.choice(
            [1, 2, 3, 4, 5],
            p=[0.05, 0.10, 0.50, 0.25, 0.10]
        )

        bonus_percent = max(0, int(np.random.normal(
            loc=performance_score * 3,
            scale=5
        )))

        # Wir erzeugen ein echtes Datum (z. B. 1. Juli des Jahres)
        date = pd.Timestamp(f"{year}-07-01")

        rows.append({
            "employee_id": emp_id,
            "date": date,
            "performance_score": performance_score,
            "bonus_percent": bonus_percent
        })

df = pd.DataFrame(rows)

# Datei speichern
df.to_csv("../data/performance_reviews.csv", index=False)

print("performance_reviews.csv wurde erfolgreich erzeugt!")
