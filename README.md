# Proyecto de Identificación de Riesgo Académico

Este proyecto utiliza el dataset `peru_student_enrollment_data_2023.csv` para identificar estudiantes con riesgo académico usando la variable objetivo `AT-RISK COURSE`.

## Estructura del proyecto

```
proyecto/
├── data/
├── outputs/
│   ├── graphs/
│   ├── models/
│   └── reports/
├── src/
│   ├── exploratory_analysis.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── main.py
├── requirements.txt
└── README.md
```

## Requisitos

- Python 3.10+.
- Instalar dependencias con `pip install -r requirements.txt`.

## Cómo ejecutar

1. Colocar `peru_student_enrollment_data_2023.csv` en `proyecto/data/`.
2. Crear y activar el entorno virtual.
3. Ejecutar:

```bash
python src/main.py
```

## Salidas generadas

- `outputs/graphs/`
  - `distribution_at_risk.png`
  - `risk_by_faculty.png`
  - `risk_by_gender.png`
  - `risk_by_age_range.png`
  - `correlation_heatmap.png`
  - `confusion_matrix.png`
  - `feature_importance.png`
- `outputs/models/random_forest_model.pkl`
- `outputs/reports/dataset_summary.txt`
- `outputs/reports/final_report.txt`

## Descripción general

El flujo del proyecto es el siguiente:

1. Cargar y analizar la estructura del dataset.
2. Generar visualizaciones EDA.
3. Limpiar valores nulos y eliminar duplicados.
4. Transformar la variable objetivo en binaria (`AT_RISK_BINARY`).
5. Codificar variables categóricas con OneHotEncoding.
6. Escalar variables numéricas relevantes.
7. Entrenar un modelo `RandomForestClassifier`.
8. Evaluar con métricas: accuracy, precision, recall, F1, confusion matrix y ROC AUC.
