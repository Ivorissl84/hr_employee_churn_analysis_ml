import pandas as pd
import numpy as np

np.random.seed(42)

# Anzahl Mitarbeiter (muss zu employees.csv passen)
n_employees = 300

# Zeitraum definieren
years = [2022, 2023, 2024]

rows = []

# Hilfsfunktionen -------------------------------------------------------------

def random_sick_days():
    """Realistische Krankheitstage pro Jahr."""
    return np.random.choice(
        [1, 3, 5, 7, 10, 14, 20],
        p=[0.15, 0.25, 0.25, 0.15, 0.10, 0.07, 0.03]
    )

def sick_blocks(total_days):
    """Krankheitstage in realistische Blöcke aufteilen."""
    blocks = []
    remaining = total_days

    while remaining > 0:
        block = np.random.randint(1, min(5, remaining) + 1)
        blocks.append(block)
        remaining -= block

    return blocks

def vacation_blocks():
    """Realistische Urlaubsblöcke für 30 Tage pro Jahr."""
    blocks = []

    # Sommerurlaub
    blocks.append(np.random.randint(10, 15))

    # Winter/Herbst
    blocks.append(np.random.randint(5, 8))

    # Brückentage
    blocks.append(np.random.randint(1, 3))

    # Rest auffüllen
    total = sum(blocks)
    remaining = 30 - total
    if remaining > 0:
        blocks.append(remaining)

    return blocks

def random_date_in_year(year):
    """Zufälliges Datum im Jahr."""
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-12-31")
    delta = (end - start).days
    return start + pd.Timedelta(days=np.random.randint(0, delta))


# Abwesenheiten erzeugen ------------------------------------------------------

for emp_id in range(1, n_employees + 1):
    for year in years:

        # -------------------------
        # 1) Urlaub (30 Tage fix)
        # -------------------------
        vac_blocks = vacation_blocks()
        for block in vac_blocks:
            date = random_date_in_year(year)
            rows.append({
                "employee_id": emp_id,
                "date": date,
                "absence_type": "vacation",
                "duration_days": block
            })

        # -------------------------
        # 2) Krankheitstage
        # -------------------------
        sick_total = random_sick_days()
        sick_blks = sick_blocks(sick_total)

        for block in sick_blks:
            date = random_date_in_year(year)
            rows.append({
                "employee_id": emp_id,
                "date": date,
                "absence_type": "sick",
                "duration_days": block
            })

        # -------------------------
        # 3) Unfall (selten)
        # -------------------------
        if np.random.rand() < 0.02:
            date = random_date_in_year(year)
            rows.append({
                "employee_id": emp_id,
                "date": date,
                "absence_type": "accident",
                "duration_days": np.random.randint(1, 4)
            })

        # -------------------------
        # 4) Other (selten)
        # -------------------------
        if np.random.rand() < 0.03:
            date = random_date_in_year(year)
            rows.append({
                "employee_id": emp_id,
                "date": date,
                "absence_type": "other",
                "duration_days": np.random.randint(1, 4)
            })

# DataFrame erzeugen
df = pd.DataFrame(rows)

# Datei speichern
df.to_csv("../data/absences.csv", index=False)

print("absences.csv wurde erfolgreich erzeugt!")
