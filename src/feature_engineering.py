import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_features(df, scale_numeric=False):
    """
    Führt das Feature Engineering für das HR‑Dataset durch:
    - Zielvariable extrahieren
    - irrelevante Spalten entfernen
    - kategorische Variablen per One‑Hot-Encoding umwandeln
    - optional numerische Variablen skalieren
    - Train/Test‑Split erzeugen
    """

    print("Starte Feature Engineering...")

    # ---------------------------------------------------
    # 1. Zielvariable definieren
    # ---------------------------------------------------
    # Sicherstellen, dass die Zielspalte vorhanden ist
    if "left_company" not in df.columns:
        raise ValueError("Spalte 'left_company' fehlt im DataFrame!")

    # Zielvariable extrahieren
    y = df["left_company"]

    # hire_date und employee_id sind für das Modell nicht relevant
    df = df.drop(columns=["hire_date", "employee_id"])

    # Feature‑Matrix ohne Zielvariable
    X = df.drop(columns=["left_company"])

    # ---------------------------------------------------
    # 2. Kategorische Variablen encoden
    # ---------------------------------------------------
    # Alle object‑Spalten identifizieren und One‑Hot‑Encoding anwenden
    categorical_cols = X.select_dtypes(include=["object"]).columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # ---------------------------------------------------
    # 3. Optional: Numerische Variablen skalieren
    # ---------------------------------------------------
    # Skalierung nur durchführen, wenn explizit aktiviert
    if scale_numeric:
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        print("Numerische Features wurden skaliert.")

    # ---------------------------------------------------
    # 4. Train/Test‑Split
    # ---------------------------------------------------
    # Stratify sorgt dafür, dass die Klassenverteilung erhalten bleibt
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Feature Engineering abgeschlossen.")
    return X_train, X_test, y_train, y_test
