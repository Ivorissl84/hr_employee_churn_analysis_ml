import pandas as pd
import numpy as np

# Reproduzierbare Zufallswerte
np.random.seed(42)

# Anzahl Mitarbeiter
n = 300

# Hilfslisten
departments = ["Production", "Logistics", "IT", "HR", "Procurement", "Sales"]
employment_types = ["Full-time", "Part-time"]

# Zufriedenheitsskala
def random_satisfaction():
    return np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.15, 0.35, 0.25, 0.15])

# Churn-Wahrscheinlichkeit (realistisch modelliert)
def generate_churn(row):
    score = 0

    # niedrige Zufriedenheit erhöht Risiko
    if row["job_satisfaction"] <= 2:
        score += 0.25
    if row["management_satisfaction"] <= 2:
        score += 0.20

    # wenig Dienstjahre → höheres Risiko
    if row["years_at_company"] < 2:
        score += 0.15

    # sehr lange Pendelzeit → höheres Risiko
    if row["commute_time_min"] > 45:
        score += 0.10

    # niedrige Performance
    if row["performance_score"] <= 2:
        score += 0.15

    # Grundrauschen
    score += np.random.uniform(0, 0.1)

    return 1 if score > 0.45 else 0


# DataFrame erzeugen
df = pd.DataFrame({
    "employee_id": range(1, n + 1),
    "gender": np.random.choice(["m", "f", "d"], size=n, p=[0.45, 0.45, 0.10]),
    "age": np.random.randint(20, 60, size=n),
    "department": np.random.choice(departments, size=n),
    "job_level": np.random.randint(1, 6, size=n),
    "employment_type": np.random.choice(employment_types, size=n, p=[0.8, 0.2]),
    "salary": np.random.randint(28000, 95000, size=n),
    "years_at_company": np.random.randint(0, 25, size=n),
    "commute_time_min": np.random.randint(5, 90, size=n),
    "job_satisfaction": [random_satisfaction() for _ in range(n)],
    "management_satisfaction": [random_satisfaction() for _ in range(n)],
    "trainings_per_year": np.random.randint(0, 6, size=n),
    "performance_score": np.random.randint(1, 6, size=n),
})

# Churn berechnen
df["left_company"] = df.apply(generate_churn, axis=1)

# Datei speichern
df.to_csv("../data/employees.csv", index=False)

print("employees.csv wurde erfolgreich mit englischen Spaltennamen erzeugt!")
