import joblib
import pandas as pd

try:
    from src.common import DEFAULT_RISK_THRESHOLD, FEATURE_NAMES, MODEL_PATH, SCALER_PATH, validate_feature_columns
except ModuleNotFoundError:  # Supports the existing ``python src/predict.py`` workflow.
    from common import DEFAULT_RISK_THRESHOLD, FEATURE_NAMES, MODEL_PATH, SCALER_PATH, validate_feature_columns


def main() -> None:
    print("🧠 Loading production AI assets...")
    try:
        model, scaler = joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
    except FileNotFoundError as exc:
        raise SystemExit(f"❌ Missing model asset: {exc.filename}. Run 'python -m src.train' first.") from exc
    raw_data = [[50000, 2, 2, 1, 35, 2, 2, 2, 2, 2, 2,
                 45000, 46000, 47000, 48000, 49000, 50000, 0, 0, 1000, 0, 0, 0]]
    customer = pd.DataFrame(raw_data, columns=FEATURE_NAMES)
    validate_feature_columns(customer)
    probability = model.predict_proba(scaler.transform(customer))[0, 1]
    print("\n--- 📊 RISK ASSESSMENT REPORT ---")
    print(f"👉 Default Probability Calculated: {probability * 100:.2f}%")
    print(f"👉 Operational Risk Threshold:     {DEFAULT_RISK_THRESHOLD * 100:.2f}%")
    print("❌ ALERT: Credit Application DENIED." if probability >= DEFAULT_RISK_THRESHOLD
          else "💚 SUCCESS: Credit Application APPROVED.")


if __name__ == "__main__":
    main()
