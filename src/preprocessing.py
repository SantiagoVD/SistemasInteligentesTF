from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Limpiar, codificar y escalar los datos para el entrenamiento de modelo."""
    df_clean = df.copy()

    # Las columnas categóricas perdidas se llenan con un valor explícito.
    categorical_cols = df_clean.select_dtypes(include='object').columns.tolist()
    df_clean[categorical_cols] = df_clean[categorical_cols].fillna('Unknown')

    # Eliminar duplicados para evitar sesgar el aprendizaje.
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    # Definir la variable objetivo binaria: 1 si el estudiante tiene al menos un curso en riesgo.
    df_clean['AT_RISK_BINARY'] = (df_clean['AT-RISK COURSE'] > 0).astype(int)

    # Selección de variables para el modelo basada en columnas reales del dataset.
    feature_columns = [
        'ENROLLMENT',
        'TUITION PAYMENT MARCH 2022',
        'TUITION PAYMENT MARCH 2023',
        'GENDER',
        'TYPE OF EDUCATIONAL INSTITUTION',
        'INSTITUTION STATUS',
        'DEPARTMENT',
        'CLASSIFICATION',
        'FACULTY',
        'PROGRAM/MAJOR',
        'SHIFT/SCHEDULE',
        'BENEFIT DISCOUNTS',
        'STUDY MODE',
        'AGE RANGE OF ENROLLED STUDENT',
        'DISABILITY',
        'NUMBER OF ENROLLED COURSES',
    ]

    X = df_clean[feature_columns].copy()
    y = df_clean['AT_RISK_BINARY'].copy()

    # Codificar categóricas con OneHotEncoding.
    categorical_features = X.select_dtypes(include='object').columns.tolist()
    X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

    # Escalamiento sólo para las columnas numéricas que lo requieren.
    numeric_columns = [
        'TUITION PAYMENT MARCH 2022',
        'TUITION PAYMENT MARCH 2023',
        'NUMBER OF ENROLLED COURSES',
    ]
    scaler = StandardScaler()
    X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

    return X, y
