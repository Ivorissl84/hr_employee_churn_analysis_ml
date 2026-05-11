import pandas as pd
import numpy as np

np.random.seed(42)

# Mitarbeiterdaten laden
employees = pd.read_csv("../data/employees.csv")

# Jahre für Bewertungen
years = [2021, 2022, 2023, 2024]

rows = []

def generate_review_score(base_score):
    """
    Erzeugt einen realistischen Performance Score für ein bestimmtes Jahr.
    Der Score schwankt leicht um den aktuellen Score aus employees.csv.
    """
    score = base_score + np.random.normal(0, 0.8)
    return int(np.clip(round(score), 1, 5))

def generate_bonus(score, department):
    """
    Bonus abhängig von Score und Department.
    Sales bekommt höhere Boni.
    """
    if department == "Sales":
        ranges = {
            1: (0, 2),
            2: (2, 4),
            3: (5, 8),
            4: (8, 12),
            5: (12, 18),
        }
    else:
        ranges = {
            1: (0, 1),
            2: (1, 3),
            3: (3, 6),
            4: (6, 10),
            5: (10, 15),
        }

    low, high = ranges[score]
    return np.random.randint(low, high + 1)

# Für jeden Mitarbeiter Bewertungen erzeugen
for _, row in employees.iterrows():
    emp_id = row["employee_id"]
    base_score = row["performance_score"]
    department = row["department"]

    # Jeder Mitarbeiter bekommt 2–4 Reviews
    n_reviews = np.random.randint(2, 5)
    review_years = np.random.choice(years, size=n_reviews, replace=False)

    for year in review_years:
        review_score = generate_review_score(base_score)
        bonus_percent = generate_bonus(review_score, department)

        date = pd.Timestamp(f"{year}-07-01")

        rows.append({
            "employee_id": emp_id,
            "date": date,
            "performance_score": review_score,
            "bonus_percent": bonus_percent
        })

df = pd.DataFrame(rows)

# Datei speichern
df.to_csv("../data/performance_reviews.csv", index=False)

print("performance_reviews.csv wurde erfolgreich erzeugt!")
