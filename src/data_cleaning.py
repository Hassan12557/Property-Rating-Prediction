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
# -------------------------------------------------------------
# 1. Category Hierarchy Splitting (Levels 1 to 5)
# -------------------------------------------------------------
category_split = df_cleaned["category"].astype(str).str.split("|", expand=True)

for i in range(5):
    col_name = f"category_level_{i+1}"
    if i in category_split.columns:
        df_cleaned[col_name] = category_split[i].fillna("Unknown").str.strip()
    else:
        df_cleaned[col_name] = "Unknown"

# -------------------------------------------------------------
# 2. Brand Extraction from product_name
# -------------------------------------------------------------
known_brands = [
    "boAt",
    "Ambrane",
    "Portronics",
    "Wayona",
    "pTron",
    "Zoul",
    "TP-Link",
    "AmazonBasics",
    "Belkin",
    "Duracell",
    "OnePlus",
    "Samsung",
    "LG",
    "MI",
    "Fire-Boltt",
    "Redmi",
    "TCL",
    "Acer",
    "Hisense",
    "VU",
    "Kodak",
]


def extract_brand(name):
    name_str = str(name).strip()
    # Check against known brand dictionary
    for brand in known_brands:
        if brand.lower() in name_str.lower():
            return brand
    # Fallback: Use first word of product name if no known brand matches
    first_word = name_str.split()[0] if len(name_str.split()) > 0 else "Unknown"
    return first_word


df_cleaned["brand"] = df_cleaned["product_name"].apply(extract_brand)

# Identify top brands vs long-tail brands
top_5_brands = df_cleaned["brand"].value_counts().nlargest(5).index.tolist()
df_cleaned["is_branded"] = (
    df_cleaned["brand"].isin(top_5_brands).astype(int)
)  #[cite: 1]

# -------------------------------------------------------------
# 3. Text Feature Extraction from about_product
# -------------------------------------------------------------
# Word count computation
df_cleaned["word_count"] = (
    df_cleaned["about_product"].astype(str).apply(lambda x: len(x.split()))
)

# Feature count heuristic (comma/pipe/bullet point counts)
df_cleaned["feature_count"] = (
    df_cleaned["about_product"]
    .astype(str)
    .apply(lambda x: x.count(",") + x.count("|") + x.count(".") + 1)
)

# Specification density calculation
df_cleaned["specification_density"] = df_cleaned["feature_count"] / (
    df_cleaned["word_count"] + 1e-5
)

# -------------------------------------------------------------
# Verification
# -------------------------------------------------------------
print("Extracted Features Summary:")
print(
    df_cleaned[
        [
            "brand",
            "is_branded",
            "category_level_1",
            "category_level_2",
            "word_count",
            "feature_count",
            "specification_density",
        ]
    ].head()
)