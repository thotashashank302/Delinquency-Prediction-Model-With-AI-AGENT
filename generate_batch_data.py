import pandas as pd
from pathlib import Path

# Resolve output path relative to this file, not the CWD
REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = REPO_ROOT / "data" / "user_uploaded_applicants.csv"

# The 23 mathematical features the model expects
feature_names = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

mock_applicants = [
    [30000, 1, 2, 2, 28,  2, 2, 2, 0, 0, 0,  28000, 29000, 27000, 22000, 20000, 19000,  0, 1000, 1500, 1000, 500, 0],
    [250000, 2, 1, 2, 31, 0, 0, 0, 0, 0, 0,  12000, 14000, 11000, 9000, 8500, 7000,  3000, 3000, 2500, 2000, 2000, 3000],
    [80000, 2, 2, 1, 45,  1, 0, 0, 0, 0, 2,  45000, 46000, 42000, 43000, 44000, 45000,  2000, 2000, 2000, 0, 1500, 0]
]

# Create the base DataFrame
df_batch = pd.DataFrame(mock_applicants, columns=feature_names)

# Insert enterprise identification data at the front
df_batch.insert(0, 'Customer_Name', ['Alex Jones', 'Sam Smith', 'Taylor Swift'])
df_batch.insert(1, 'Email', ['alex.jones@gmail.com', 'sam.smith@yahoo.com', 'taylor.swift@reputation.com'])

# Ensure output directory exists and save
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_batch.to_csv(OUTPUT_PATH, index=False)
print(f"📁 Production test file created at: {OUTPUT_PATH}")