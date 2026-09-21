import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from src.common import (
        DEFAULT_RISK_THRESHOLD,
        EVALUATION_PATH,
        FEATURE_NAMES,
        METADATA_PATH,
        MODEL_PATH,
        RANDOM_STATE,
        RAW_DATA_PATH,
        SCALER_PATH,
    )
except ModuleNotFoundError:  # Supports the existing ``python src/train.py`` workflow.
    from common import (
        DEFAULT_RISK_THRESHOLD,
        EVALUATION_PATH,
        FEATURE_NAMES,
        METADATA_PATH,
        MODEL_PATH,
        RANDOM_STATE,
        RAW_DATA_PATH,
        SCALER_PATH,
    )


def main() -> None:
    print("🔄 Loading clean dataset...")
    df = pd.read_excel(RAW_DATA_PATH, header=1)
    X, y = df.loc[:, FEATURE_NAMES], df["default payment next month"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print("⚖️ Scaling features...")
    scaler = StandardScaler()
    X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)
    print("\n🌲 Training production Random Forest Model...")
    model = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    print("✅ Model Training Complete!")
    probabilities = model.predict_proba(X_test_scaled)[:, 1]
    y_pred_custom = (probabilities >= DEFAULT_RISK_THRESHOLD).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred_custom),
        "average_precision": average_precision_score(y_test, probabilities),
        "precision": precision_score(y_test, y_pred_custom, zero_division=0),
        "recall": recall_score(y_test, y_pred_custom, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "confusion_matrix": confusion_matrix(y_test, y_pred_custom).tolist(),
        "threshold": DEFAULT_RISK_THRESHOLD,
        "test_rows": len(y_test),
    }
    print(f"📊 Accuracy: {metrics['accuracy']:.3f}")
    print(f"📊 ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"📊 Recall:  {metrics['recall']:.3f}")
    print(classification_report(y_test, y_pred_custom, zero_division=0))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    EVALUATION_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "RandomForestClassifier",
        "feature_count": len(FEATURE_NAMES),
        "features": list(FEATURE_NAMES),
        "training_rows": len(X_train),
        "random_state": RANDOM_STATE,
        "sklearn_class_weight": "balanced",
        "n_estimators": 100,
        "risk_threshold": DEFAULT_RISK_THRESHOLD,
        "dataset": RAW_DATA_PATH.name,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"🎉 Saved model artifacts to {MODEL_PATH.parent}")


if __name__ == "__main__":
    main()
