import pandas as pd
import numpy as np

# Load raw Amazon dataset
df = pd.read_csv("D:/Data Science Projects/Property-Rating-Prediction/Data/amazon.csv")

# Define columns to drop based on pre-launch constraints and non-predictive noise
unnecessary_columns = [
    "product_id",
    "user_id",
    "user_name",
    "review_id",
    "review_title",
    "review_content",
    "img_link",
    "product_link",
]

# Drop unnecessary columns if they exist in the dataframe
df_filtered = df.drop(
    columns=[col for col in unnecessary_columns if col in df.columns]
)

# Verify retained schema
print("Remaining Columns:", df_filtered.columns.tolist())
print("Dataset Shape:", df_filtered.shape)
# 1. Standard pandas null count
print("=== Explicit Null Counts ===")
print(df_filtered.isnull().sum())

# 2. Identify the specific row with the invalid rating '|'
invalid_rating_mask = ~df_filtered["rating"].str.replace(".", "", regex=False).str.isdigit()
invalid_rating_rows = df_filtered[invalid_rating_mask]

print("\n=== Rows with Non-Numeric Rating ===")
print(invalid_rating_rows[["product_name", "rating", "rating_count"]])

# 3. Identify rows with missing rating_count
missing_rating_count_rows = df_filtered[df_filtered["rating_count"].isnull()]

print("\n=== Rows with Missing rating_count ===")
print("=== NULL VALUE CHECK BEFORE CLEANING ===")
print(df_filtered.isnull().sum())
print("\n")

# 3. Clean String Formatting & Cast to Numeric Types


def clean_currency_and_numbers(series):
    """Removes currency symbols (₹), commas, and percentage signs."""
    return (
        series.astype(str)
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )


# Clean price and count columns
df_filtered["discounted_price"] = pd.to_numeric(
    clean_currency_and_numbers(df_filtered["discounted_price"]),
    errors="coerce",
)

df_filtered["actual_price"] = pd.to_numeric(
    clean_currency_and_numbers(df_filtered["actual_price"]), errors="coerce"
)

df_filtered["discount_percentage"] = pd.to_numeric(
    clean_currency_and_numbers(df_filtered["discount_percentage"]),
    errors="coerce",
)

df_filtered["rating_count"] = pd.to_numeric(
    clean_currency_and_numbers(df_filtered["rating_count"]), errors="coerce"
)

# Clean target variable (handles non-numeric characters like '|')
df_filtered["rating"] = pd.to_numeric(
    df_filtered["rating"].astype(str).str.strip(), errors="coerce"
)


# 4. Handle Missing / Null Values (Imputation Strategy)

# A. Drop rows missing the target variable 'rating'
df_cleaned = df_filtered.dropna(subset=["rating"]).copy()

# B. Impute missing numerical predictors using median
if df_cleaned["rating_count"].isnull().sum() > 0:
    median_rating_count = df_cleaned["rating_count"].median()
    df_cleaned["rating_count"] = df_cleaned["rating_count"].fillna(
        median_rating_count
    )

if df_cleaned["discounted_price"].isnull().sum() > 0:
    df_cleaned["discounted_price"] = df_cleaned["discounted_price"].fillna(
        df_cleaned["discounted_price"].median()
    )

if df_cleaned["actual_price"].isnull().sum() > 0:
    df_cleaned["actual_price"] = df_cleaned["actual_price"].fillna(
        df_cleaned["actual_price"].median()
    )

# C. Impute missing text columns with default strings
df_cleaned["about_product"] = df_cleaned["about_product"].fillna("Unknown")
df_cleaned["category"] = df_cleaned["category"].fillna("Unknown")
df_cleaned["product_name"] = df_cleaned["product_name"].fillna("Unknown")


# 5. Verify Cleaned Dataset State
print("=== NULL VALUE CHECK AFTER CLEANING & IMPUTATION ===")
print(df_cleaned.isnull().sum())
print("\nDataset Shape After Cleaning:", df_cleaned.shape)
print("Data Types:\n", df_cleaned.dtypes)