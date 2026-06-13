from pathlib import Path

from sklearn.model_selection import train_test_split

from exploratory_analysis import load_data, perform_eda
from evaluate_model import evaluate_model
from preprocessing import preprocess_data
from train_model import train_random_forest


def main() -> None:
    """Punto de entrada principal del proyecto de identificación de riesgo académico."""
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / 'data' / 'peru_student_enrollment_data_2023.csv'
    graph_dir = project_root / 'outputs' / 'graphs'
    model_dir = project_root / 'outputs' / 'models'
    report_dir = project_root / 'outputs' / 'reports'

    if not data_path.exists():
        raise FileNotFoundError(
            f'Dataset no encontrado en {data_path}. Coloca peru_student_enrollment_data_2023.csv en el directorio data.'
        )

    print('Cargando datos...')
    df = load_data(data_path)

    print('Realizando análisis exploratorio (EDA)...')
    perform_eda(df, graph_dir, report_dir)

    print('Preprocesando datos...')
    X, y = preprocess_data(df)

    print('Dividiendo datos en entrenamiento y prueba...')
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    print('Entrenando el modelo RandomForest...')
    model_path = model_dir / 'random_forest_model.pkl'
    model = train_random_forest(X_train, y_train, model_path)

    print('Evaluando el modelo...')
    report_path = report_dir / 'final_report.txt'
    metrics = evaluate_model(model, X_test, y_test, graph_dir, report_path)

    print('Entrenamiento y evaluación completados.')
    print(f'Modelo guardado en: {model_path}')
    print(f'Gráficas guardadas en: {graph_dir}')
    print(f'Reportes guardados en: {report_path}')
    print(f'Métricas principales: {metrics}')


if __name__ == '__main__':
    main()
