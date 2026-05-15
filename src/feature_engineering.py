import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler


def prepare_features(df, scale_numeric=False):
    """
    Führt das Feature Engineering für das HR‑Dataset durch – ohne Prints.
    Gibt zurück:
    - X_train_resampled
    - X_test
    - y_train_resampled
    - y_test
    - employee_ids_train
    - employee_ids_test
    """

    # ---------------------------------------------------
    # 0. employee_id sichern
    # ---------------------------------------------------
    if "employee_id" not in df.columns:
        raise ValueError("Spalte 'employee_id' fehlt im DataFrame!")

    employee_ids = df["employee_id"].copy()

    # ---------------------------------------------------
    # 1. Zielvariable extrahieren
    # ---------------------------------------------------
    if "left_company" not in df.columns:
        raise ValueError("Spalte 'left_company' fehlt im DataFrame!")

    y = df["left_company"]

    # hire_date und employee_id entfernen
    df = df.drop(columns=["hire_date", "employee_id"])

    # Feature‑Matrix ohne Zielvariable
    X = df.drop(columns=["left_company"])

    # ---------------------------------------------------
    # 2. One‑Hot‑Encoding
    # ---------------------------------------------------
    categorical_cols = X.select_dtypes(include=["object"]).columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # ---------------------------------------------------
    # 3. Optional: Skalierung
    # ---------------------------------------------------
    if scale_numeric:
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # ---------------------------------------------------
    # 4. Train/Test‑Split (inkl. employee_id)
    # ---------------------------------------------------
    X_train, X_test, y_train, y_test, employee_ids_train, employee_ids_test = train_test_split(
        X, y, employee_ids,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ---------------------------------------------------
    # 5. Oversampling
    # ---------------------------------------------------
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

    # ---------------------------------------------------
    # 6. Rückgabe
    # ---------------------------------------------------
    return (
        X_train_resampled,
        X_test,
        y_train_resampled,
        y_test,
        employee_ids_train,
        employee_ids_test
    )
