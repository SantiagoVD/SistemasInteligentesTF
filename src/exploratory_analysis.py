from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_data(csv_path: Path) -> pd.DataFrame:
    """Cargar el dataset usando el delimitador real del CSV."""
    return pd.read_csv(csv_path, delimiter=';', encoding='utf-8')


def save_dataset_summary(df: pd.DataFrame, report_path: Path) -> None:
    """Guardar un reporte de la estructura del dataset en un archivo de texto."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open('w', encoding='utf-8') as report:
        report.write('DATASET SUMMARY\n')
        report.write('================\n')
        report.write(f'Rows: {df.shape[0]}\n')
        report.write(f'Columns: {df.shape[1]}\n\n')
        report.write('Column types:\n')
        report.write(df.dtypes.astype(str).to_string())
        report.write('\n\nNull values by column:\n')
        report.write(df.isnull().sum().to_string())
        report.write('\n\nDuplicate rows:\n')
        report.write(str(df.duplicated().sum()) + '\n\n')
        report.write('Descriptive statistics (all columns):\n')
        report.write(df.describe(include='all').transpose().to_string())


def plot_distribution_at_risk(df: pd.DataFrame, output_file: Path) -> None:
    """Generar y guardar la distribución de alumnos en riesgo académico."""
    df_plot = df.copy()
    df_plot['AT_RISK_BINARY'] = (df_plot['AT-RISK COURSE'] > 0).astype(int)
    counts = df_plot['AT_RISK_BINARY'].value_counts().sort_index()
    labels = ['No Risk', 'At Risk']

    plt.figure(figsize=(8, 6))
    sns.barplot(x=labels, y=counts.values, hue=labels, palette='muted', dodge=False, legend=False)
    plt.xlabel('Academic Risk')
    plt.ylabel('Number of Students')
    plt.title('Distribución de estudiantes por riesgo académico')
    for index, value in enumerate(counts.values):
        plt.text(index, value + counts.values.max() * 0.01, str(value), ha='center')
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200)
    plt.close()


def plot_risk_by_category(df: pd.DataFrame, category: str, output_file: Path) -> None:
    """Generar y guardar una gráfica de riesgo académico agrupada por una categoría."""
    df_plot = df.copy()
    df_plot['AT_RISK_BINARY'] = (df_plot['AT-RISK COURSE'] > 0).astype(int)
    group = df_plot.groupby(category)['AT_RISK_BINARY'].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=group.values, y=group.index, hue=group.index, palette='coolwarm', dodge=False, legend=False)
    plt.xlabel('Proporción de estudiantes en riesgo')
    plt.ylabel(category)
    plt.title(f'Riesgo académico por {category}')
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, output_file: Path) -> None:
    """Generar y guardar un mapa de calor de correlaciones entre variables numéricas."""
    df_numeric = df[['TUITION PAYMENT MARCH 2022', 'TUITION PAYMENT MARCH 2023', 'NUMBER OF ENROLLED COURSES', 'AT-RISK COURSE']].copy()
    df_numeric['AT_RISK_BINARY'] = (df_numeric['AT-RISK COURSE'] > 0).astype(int)
    corr = df_numeric.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', square=True, cbar_kws={'shrink': 0.75})
    plt.title('Heatmap de correlaciones')
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200)
    plt.close()


def perform_eda(df: pd.DataFrame, graph_dir: Path, report_dir: Path) -> None:
    """Ejecutar el análisis exploratorio completo y guardar los resultados."""
    report_file = report_dir / 'dataset_summary.txt'
    save_dataset_summary(df, report_file)
    plot_distribution_at_risk(df, graph_dir / 'distribution_at_risk.png')
    plot_risk_by_category(df, 'FACULTY', graph_dir / 'risk_by_faculty.png')
    plot_risk_by_category(df, 'GENDER', graph_dir / 'risk_by_gender.png')
    plot_risk_by_category(df, 'AGE RANGE OF ENROLLED STUDENT', graph_dir / 'risk_by_age_range.png')
    plot_correlation_heatmap(df, graph_dir / 'correlation_heatmap.png')
