import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from src.common import DEFAULT_RISK_THRESHOLD, FEATURE_NAMES, MODEL_PATH, RAW_DATA_PATH, SCALER_PATH
except ModuleNotFoundError:  # Supports the existing ``python src/train.py`` workflow.
    from common import DEFAULT_RISK_THRESHOLD, FEATURE_NAMES, MODEL_PATH, RAW_DATA_PATH, SCALER_PATH


def main() -> None:
    print("🔄 Loading clean dataset...")
    df = pd.read_excel(RAW_DATA_PATH, header=1)
    X, y = df.loc[:, FEATURE_NAMES], df["default payment next month"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("⚖️ Scaling features...")
    scaler = StandardScaler()
    X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)
    print("\n🌲 Training production Random Forest Model...")
    model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    model.fit(X_train_scaled, y_train)
    print("✅ Model Training Complete!")
    probabilities = model.predict_proba(X_test_scaled)[:, 1]
    y_pred_custom = (probabilities >= DEFAULT_RISK_THRESHOLD).astype(int)
    print(f"📊 New Accuracy Score: {accuracy_score(y_test, y_pred_custom) * 100:.2f}%")
    print(classification_report(y_test, y_pred_custom))
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"🎉 Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
