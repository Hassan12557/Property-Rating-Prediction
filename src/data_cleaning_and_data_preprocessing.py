import numpy as np
import pandas as pd

# 1. Load raw Amazon dataset
df = pd.read_csv(
    "D:/Data Science Projects/Property-Rating-Prediction/Data/amazon.csv"
)

# 2. Define columns to drop based on pre-launch constraints
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
).copy()


# 3. Helper function to clean currency symbols and formatting
def clean_currency_and_numbers(series: pd.Series) -> pd.Series:
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
    clean_currency_and_numbers(df_filtered["actual_price"]),
    errors="coerce",
)

df_filtered["discount_percentage"] = pd.to_numeric(
    clean_currency_and_numbers(df_filtered["discount_percentage"]),
    errors="coerce",
)

df_filtered["rating_count"] = pd.to_numeric(
    clean_currency_and_numbers(df_filtered["rating_count"]),
    errors="coerce",
)

# Clean target variable (handles non-numeric characters like '|')
df_filtered["rating"] = pd.to_numeric(
    df_filtered["rating"].astype(str).str.strip(),
    errors="coerce",
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


# 5. Category Hierarchy Splitting (Levels 1 to 5)
category_split = df_cleaned["category"].astype(str).str.split("|", expand=True)

for i in range(5):
    col_name = f"category_level_{i+1}"
    if i in category_split.columns:
        df_cleaned[col_name] = category_split[i].fillna("Unknown").str.strip()
    else:
        df_cleaned[col_name] = "Unknown"


# 6. Brand Extraction from product_name
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


def extract_brand(name: str) -> str:
    name_str = str(name).strip()
    for brand in known_brands:
        if brand.lower() in name_str.lower():
            return brand
    first_word = name_str.split()[0] if len(name_str.split()) > 0 else "Unknown"
    return first_word


df_cleaned["brand"] = df_cleaned["product_name"].apply(extract_brand)

# Identify top 5 brands vs others
top_5_brands = df_cleaned["brand"].value_counts().nlargest(5).index.tolist()
df_cleaned["is_branded"] = df_cleaned["brand"].isin(top_5_brands).astype(int)


# 7. Text Feature Extraction from about_product
df_cleaned["word_count"] = (
    df_cleaned["about_product"].astype(str).apply(lambda x: len(x.split()))
)

df_cleaned["feature_count"] = (
    df_cleaned["about_product"]
    .astype(str)
    .apply(lambda x: x.count(",") + x.count("|") + x.count(".") + 1)
)

df_cleaned["specification_density"] = df_cleaned["feature_count"] / (
    df_cleaned["word_count"] + 1e-5
)


# 8. Log Transformations for Skewed Features
df_cleaned["log_price"] = np.log1p(df_cleaned["discounted_price"])
df_cleaned["log_actual_price"] = np.log1p(df_cleaned["actual_price"])
df_cleaned["popularity_score"] = np.log1p(df_cleaned["rating_count"])


# 9. Derived Financial Features
df_cleaned["price_premium"] = (
    df_cleaned["actual_price"] - df_cleaned["discounted_price"]
)
df_cleaned["price_ratio"] = df_cleaned["discounted_price"] / (
    df_cleaned["actual_price"] + 1e-5
)


# 10. Category Relative Indices (Grouped by Category Level 1)
cat_median_price = df_cleaned.groupby("category_level_1")[
    "discounted_price"
].transform("median")
cat_mean_discount = df_cleaned.groupby("category_level_1")[
    "discount_percentage"
].transform("mean")

df_cleaned["category_price_index"] = df_cleaned["discounted_price"] / (
    cat_median_price + 1e-5
)
df_cleaned["category_discount_index"] = df_cleaned["discount_percentage"] / (
    cat_mean_discount + 1e-5
)


# 11. Verification Output
print("=== PREPROCESSING & FEATURE ENGINEERING COMPLETE ===")
print("Dataset Shape:", df_cleaned.shape)
print("\n=== SKEWNESS REDUCTION VERIFICATION ===")
print(f"Raw discounted_price Skew: {df_cleaned['discounted_price'].skew():.2f}")
print(f"Log discounted_price Skew: {df_cleaned['log_price'].skew():.2f}")
print(f"Raw rating_count Skew:     {df_cleaned['rating_count'].skew():.2f}")
print(f"Log popularity_score Skew: {df_cleaned['popularity_score'].skew():.2f}")

# Save your cleaned dataset for subsequent modules
df_cleaned.to_csv(
    "D:/Data Science Projects/Property-Rating-Prediction/Data/amazon_cleaned.csv",
    index=False,
)