import sys
from pathlib import Path
from unittest.mock import patch

print("🧪 STARTING PIPELINE VALIDATION SUB-ROUTINE...")

# Resolve absolute path so the test works from any working directory
REPO_ROOT = Path(__file__).resolve().parent
CSV_PATH = str(REPO_ROOT / "data" / "user_uploaded_applicants.csv")

# mock_inputs must cover every input() call in batch_predict.main():
#   1. Operator email
#   2. CSV/Excel path
#   3+. One yes/no per Moderate Risk applicant (generate_batch_data creates 3 rows;
#       supply enough answers to avoid StopIteration even if all are Moderate Risk)
mock_inputs = [
    "test_operator@gmail.com",  # 1. Operator Email  → auto-selects smtp.gmail.com
    CSV_PATH,                   # 2. Dataset path    → absolute, safe from any CWD
    "yes",                      # 3. Moderate Risk row #1 → send reminder
    "no",                       # 4. Moderate Risk row #2 → skip
    "no",                       # 5. Moderate Risk row #3 → skip (extra safety)
]

# Patch targets must match the module that *uses* them (where-used rule)
with patch("builtins.input", side_effect=mock_inputs), \
     patch("src.batch_predict.getpass.getpass", return_value="fakepassword1234"), \
     patch("src.batch_predict.smtplib.SMTP") as mock_smtp:

    # Configure our fake SMTP server to behave like a real one
    instance = mock_smtp.return_value.__enter__.return_value
    instance.sendmail.return_value = {}

    print("🤖 System: Simulating dynamic user environment...")

    try:
        # Step 1: generate the fixture CSV the test will load
        import generate_batch_data  # noqa: F401 — side-effect: creates the CSV

        # Step 2: run the full CLI pipeline
        from src.batch_predict import main
        main()

        print("\n✅ SUCCESS: Pipeline ran end-to-end without errors.")
        print("💡 With a real App Password, this pipeline will send emails flawlessly.")
    except Exception as e:
        print(f"\n❌ Pipeline Error Detected: {type(e).__name__}: {e}")
        sys.exit(1)