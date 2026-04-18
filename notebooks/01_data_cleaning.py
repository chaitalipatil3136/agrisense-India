import pandas as pd
import os

print("STEP 1: Script started")

# Get project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "india_crop_production.csv")
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "master_dataset.csv")

print("STEP 2: Loading dataset from:", RAW_PATH)

# Load dataset
df = pd.read_csv(RAW_PATH, encoding="latin-1")

print("STEP 3: Dataset loaded")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# Clean column names
df.columns = [col.strip().lower() for col in df.columns]

# Rename columns
df = df.rename(columns={
    "state_name": "state",
    "district_name": "district",
    "crop": "crop",
    "season": "season",
    "year": "year",
    "area": "area",
    "production": "production"
})

# Missing values
print("\nSTEP 4: Missing values check")
print(df.isnull().sum())

# Handle missing production
if df["production"].isnull().sum() > 0:
    median_val = df["production"].median()
    df["production"].fillna(median_val, inplace=True)
    print("Production missing values imputed")

# Remove duplicates
df.drop_duplicates(inplace=True)

# Save processed file
os.makedirs(os.path.join(BASE_DIR, "data", "processed"), exist_ok=True)
df.to_csv(PROCESSED_PATH, index=False)

print("\nSTEP 5: Data cleaning complete")
print("Saved to:", PROCESSED_PATH)
print("Final shape:", df.shape)