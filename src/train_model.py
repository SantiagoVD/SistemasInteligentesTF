from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier


def train_random_forest(X_train, y_train, model_path: Path) -> RandomForestClassifier:
    """Entrenar un RandomForestClassifier y guardar el modelo en disco."""
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model
