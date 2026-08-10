import pandas as pd

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
