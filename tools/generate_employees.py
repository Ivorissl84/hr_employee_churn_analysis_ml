import pandas as pd
import numpy as np

# ---------------------------------------------------
# Reproduzierbare Zufallswerte
# ---------------------------------------------------
np.random.seed(42)

# ---------------------------------------------------
# Anzahl Mitarbeiter
# ---------------------------------------------------
n = 300

# ---------------------------------------------------
# Hilfslisten
# ---------------------------------------------------
departments = ["Production", "Logistics", "IT", "HR", "Procurement", "Sales"]
employment_types = ["Full-time", "Part-time"]

# ---------------------------------------------------
# Zufriedenheitsskala (realistische Verteilung)
# ---------------------------------------------------
def random_satisfaction():
    return np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.15, 0.35, 0.25, 0.15])

# ---------------------------------------------------
# Realistische Gehaltsbänder pro Department
# ---------------------------------------------------
salary_bands = {
    "Production": (30000, 48000),
    "Logistics": (28000, 45000),
    "HR": (35000, 55000),
    "IT": (38000, 58000),
    "Sales": (40000, 70000),
    "Procurement": (45000, 75000),
}

# ---------------------------------------------------
# Gehalt generieren
# ---------------------------------------------------
def generate_salary(dept, job_level, employment_type, years):
    base_min, base_max = salary_bands[dept]
    base = np.random.randint(base_min, base_max)

    # Job Level wirkt als Multiplikator
    level_factor = 1 + (job_level - 1) * 0.1
    salary = base * level_factor

    # Erfahrung wirkt leicht positiv
    salary *= (1 + years * 0.01)

    # Part-time reduziert Gehalt
    if employment_type == "Part-time":
        salary *= np.random.uniform(0.5, 0.8)

    return int(salary)

# ---------------------------------------------------
# Performance Score realistisch generieren
# ---------------------------------------------------
def generate_performance(job_sat, mgmt_sat, trainings, job_level):
    score = 3  # Basiswert

    if job_sat >= 4:
        score += 1
    if mgmt_sat >= 4:
        score += 1
    if trainings >= 3:
        score += 1

    # Höhere Job Level leicht besser
    score += (job_level - 3) * 0.2

    # Zufall für Realismus
    score += np.random.normal(0, 0.5)

    return int(np.clip(round(score), 1, 5))

# ---------------------------------------------------
# Realistische Churn-Wahrscheinlichkeit (Option B)
# ---------------------------------------------------
def generate_churn(row):
    score = 0

    # --- Unzufriedenheit ---
    if row["job_satisfaction"] <= 2:
        score += 0.35
    if row["management_satisfaction"] <= 2:
        score += 0.30

    # --- Kurze Betriebszugehörigkeit ---
    if row["years_at_company"] < 2:
        score += 0.20

    # --- Lange Pendelzeit ---
    if row["commute_time_min"] > 45:
        score += 0.15

    # --- Schlechte Performance ---
    if row["performance_score"] <= 2:
        score += 0.25

    # --- Department-Risiko ---
    dept_risk = {
        "Production": 0.05,
        "Logistics": 0.10,
        "IT": 0.20,
        "HR": 0.08,
        "Procurement": 0.04,
        "Sales": 0.18
    }
    score += dept_risk[row["department"]]

    # --- Job Level Risiko ---
    if row["job_level"] == 1:
        score += 0.15
    elif row["job_level"] == 2:
        score += 0.10

    # --- Trainings reduzieren Risiko ---
    if row["trainings_per_year"] >= 3:
        score -= 0.10

    # --- Zufällige Varianz ---
    score += np.random.uniform(0, 0.15)

    # --- Finaler Churn-Entscheid ---
    return 1 if score > 0.55 else 0

# ---------------------------------------------------
# Grunddaten erzeugen
# ---------------------------------------------------
df = pd.DataFrame({
    "employee_id": range(1, n + 1),
    "gender": np.random.choice(["m", "f", "d"], size=n, p=[0.45, 0.45, 0.10]),
    "age": np.random.randint(20, 60, size=n),
    "department": np.random.choice(departments, size=n),
    "job_level": np.random.randint(1, 6, size=n),
    "employment_type": np.random.choice(employment_types, size=n, p=[0.8, 0.2]),
    "years_at_company": np.random.randint(0, 25, size=n),
    "commute_time_min": np.random.randint(5, 90, size=n),
    "job_satisfaction": [random_satisfaction() for _ in range(n)],
    "management_satisfaction": [random_satisfaction() for _ in range(n)],
    "trainings_per_year": np.random.randint(0, 6, size=n),
})

# ---------------------------------------------------
# Performance Score generieren
# ---------------------------------------------------
df["performance_score"] = df.apply(
    lambda row: generate_performance(
        row["job_satisfaction"],
        row["management_satisfaction"],
        row["trainings_per_year"],
        row["job_level"]
    ),
    axis=1
)

# ---------------------------------------------------
# Gehalt generieren
# ---------------------------------------------------
df["salary"] = df.apply(
    lambda row: generate_salary(
        row["department"],
        row["job_level"],
        row["employment_type"],
        row["years_at_company"]
    ),
    axis=1
)

# ---------------------------------------------------
# Churn berechnen (neue Logik)
# ---------------------------------------------------
df["left_company"] = df.apply(generate_churn, axis=1)

# ---------------------------------------------------
# Datei speichern
# ---------------------------------------------------
df.to_csv("../data/employees.csv", index=False)
print("employees.csv wurde erfolgreich erzeugt!")
