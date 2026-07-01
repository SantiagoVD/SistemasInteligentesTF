from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier


def train_random_forest(X_train, y_train, model_path: Path) -> RandomForestClassifier:
    """Entrenar un RandomForestClassifier y guardar el modelo en disco."""
    def create_model():
    return RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model = create_model()
    model.fit(X_train, y_train)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model
