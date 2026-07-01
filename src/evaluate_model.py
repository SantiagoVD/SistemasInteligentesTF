from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)


def save_confusion_matrix(y_true, y_pred, output_file: Path) -> None:
    """Guardar la matriz de confusión como una imagen."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200)
    plt.close()


def save_feature_importance(model, feature_names, output_file: Path, top_n: int = 20) -> None:
    """Guardar la importancia de features del modelo entrenado."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    plt.figure(figsize=(9, 7))
    sns.barplot(x=top_importances, y=top_features, palette='viridis')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance (top {} variables)'.format(top_n))
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200)
    plt.close()


def evaluate_model(model, X_test, y_test, graph_dir: Path, report_path: Path) -> dict:
    """Evaluar el modelo y guardar métricas, reporte y gráficas de evaluación."""
    y_pred = model.predict(X_test)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
    }

    y_prob = None
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
        try:
            metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
        except ValueError:
            metrics['roc_auc'] = None

    save_confusion_matrix(y_test, y_pred, graph_dir / 'confusion_matrix.png')
    save_feature_importance(model, X_test.columns.tolist(), graph_dir / 'feature_importance.png')

    with report_path.open('w', encoding='utf-8') as report:
        report.write('MODEL EVALUATION REPORT\n')
        report.write('========================\n')
        report.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        report.write(f"Precision: {metrics['precision']:.4f}\n")
        report.write(f"Recall: {metrics['recall']:.4f}\n")
        report.write(f"F1 Score: {metrics['f1_score']:.4f}\n")
        report.write(f"ROC AUC: {metrics.get('roc_auc', None):.4f}\n" if metrics.get('roc_auc') is not None else 'ROC AUC: N/A\n')
        report.write('\nClassification Report:\n')
        report.write(classification_report(y_test, y_pred, zero_division=0))

    return metrics
