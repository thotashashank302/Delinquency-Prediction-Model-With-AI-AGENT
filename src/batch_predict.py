import getpass
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    from src.common import (
        FEATURE_NAMES,
        MODEL_PATH,
        REPO_ROOT,
        SCALER_PATH,
        classify_risk,
        validate_feature_columns,
    )
except ModuleNotFoundError:  # Supports the existing ``python src/batch_predict.py`` workflow.
    from common import (
        FEATURE_NAMES,
        MODEL_PATH,
        REPO_ROOT,
        SCALER_PATH,
        classify_risk,
        validate_feature_columns,
    )


def prepare_report(frame: pd.DataFrame, model, scaler) -> pd.DataFrame:
    """Validate applicants and create predictions without email side effects."""
    validate_feature_columns(frame, require_email=True)
    probabilities = model.predict_proba(scaler.transform(frame.loc[:, FEATURE_NAMES]))[:, 1]
    report = frame.copy()
    report["Delinquency_Probability_%"] = np.round(probabilities * 100, 2)
    report["Assessed_Risk_Level"] = [classify_risk(value) for value in probabilities]
    return report


def send_reminder(
    sender_email: str, sender_password: str, smtp_server: str, customer_email: str
) -> None:
    msg = MIMEMultipart()
    msg["From"], msg["To"], msg["Subject"] = (
        sender_email,
        customer_email,
        "Urgent Account Notice: Financial Health Check-In",
    )
    msg.attach(
        MIMEText(
            "Dear Customer,\n\nPlease review your outstanding balance statements."
            "\n\nBest Regards,\nCredit Risk Automation Desk",
            "plain",
        )
    )
    with smtplib.SMTP(smtp_server, 587) as server:
        server.starttls(context=ssl.create_default_context())
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, customer_email, msg.as_string())


def main() -> None:
    try:
        model, scaler = joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"❌ Missing model asset: {exc.filename}. Run 'python -m src.train' first."
        ) from exc
    sender_email = input("✉️ Enter YOUR email address (to send from): ").strip()
    sender_password = getpass.getpass("🔑 Enter your App Password (hidden): ").strip()
    if "@gmail.com" in sender_email.lower():
        smtp_server = "smtp.gmail.com"
    elif any(domain in sender_email.lower() for domain in ("@outlook.com", "@hotmail.com")):
        smtp_server = "smtp.office365.com"
    else:
        smtp_server = input("🌐 Enter your SMTP server: ").strip()
    input_path = Path(input("📥 Enter your CSV/Excel path: ").strip().strip("'\" "))
    if not input_path.is_file():
        raise SystemExit(f"❌ Error: Could not find file at '{input_path}'.")
    try:
        suffix = input_path.suffix.lower()
        if suffix in (".xlsx", ".xls"):
            frame = pd.read_excel(input_path)
        elif suffix == ".csv":
            frame = pd.read_csv(input_path)
        else:
            raise SystemExit(
                f"❌ Unsupported file type '{suffix}'. "
                "Please provide a .csv or .xlsx file.\n"
                "Tip: In Apple Numbers, go to File → Export To → CSV to convert .numbers files."
            )
        report = prepare_report(frame, model, scaler)
    except (pd.errors.ParserError, ValueError, UnicodeDecodeError) as exc:
        raise SystemExit(f"❌ Validation Error: {exc}") from exc
    print(
        report[["Email", "Delinquency_Probability_%", "Assessed_Risk_Level"]].to_string(index=False)
    )
    for _, row in report[report["Assessed_Risk_Level"] == "Moderate Risk"].iterrows():
        approved = input(f"❓ Send a reminder to {row['Email']}? (yes/no): ")
        if approved.strip().lower() in {"yes", "y"}:
            try:
                send_reminder(sender_email, sender_password, smtp_server, row["Email"])
            except (OSError, smtplib.SMTPException) as exc:
                print(f"❌ Mail Error: {exc}")
    output_dir = REPO_ROOT / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    if suffix in (".xlsx", ".xls"):
        output_path = output_dir / f"evaluated_{input_path.stem}.xlsx"
        report.to_excel(output_path, index=False)
    else:
        output_path = output_dir / f"evaluated_{input_path.stem}.csv"
        report.to_csv(output_path, index=False)
    print(f"💾 Evaluated file saved to: {output_path}")


if __name__ == "__main__":
    main()
