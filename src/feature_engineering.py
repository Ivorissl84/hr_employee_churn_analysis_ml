import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

def prepare_features(df, scale_numeric=False):
    """
    Führt das Feature Engineering für das HR‑Dataset durch:
    - Zielvariable extrahieren
    - irrelevante Spalten entfernen
    - kategorische Variablen per One‑Hot-Encoding umwandeln
    - optional numerische Variablen skalieren
    - Train/Test‑Split erzeugen
    - Oversampling der Minderheitsklasse (Option A)
    """

    print("Starte Feature Engineering...")

    # ---------------------------------------------------
    # 1. Zielvariable definieren
    # ---------------------------------------------------
    if "left_company" not in df.columns:
        raise ValueError("Spalte 'left_company' fehlt im DataFrame!")

    y = df["left_company"]

    # hire_date und employee_id sind für das Modell nicht relevant
    df = df.drop(columns=["hire_date", "employee_id"])

    # Feature‑Matrix ohne Zielvariable
    X = df.drop(columns=["left_company"])

    # ---------------------------------------------------
    # 2. Kategorische Variablen encoden
    # ---------------------------------------------------
    categorical_cols = X.select_dtypes(include=["object"]).columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # ---------------------------------------------------
    # 3. Optional: Numerische Variablen skalieren
    # ---------------------------------------------------
    if scale_numeric:
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        print("Numerische Features wurden skaliert.")

    # ---------------------------------------------------
    # 4. Train/Test‑Split
    # ---------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------------------------------------------------
    # 5. Oversampling der Minderheitsklasse (Option A)
    # ---------------------------------------------------
    # RandomOverSampler ist stabiler als SMOTE bei gemischten Datentypen
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

    print("Oversampling abgeschlossen.")
    print("Feature Engineering abgeschlossen.")

    return X_train_resampled, X_test, y_train_resampled, y_test
