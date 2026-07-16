import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load cleaned dataset
df = pd.read_parquet("data/processed/cleaned_dataset.parquet")

# Separate features and target
X = df.drop("Label", axis=1)
y = df["Label"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# Find categorical columns
categorical_columns = X.select_dtypes(include=["object", "category"]).columns

print("Categorical Columns:")
print(categorical_columns)

# Apply One-Hot Encoding
X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True
)

print("\nShape after One-Hot Encoding:", X.shape)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nEncoded Classes:")
for i, cls in enumerate(label_encoder.classes_):
    print(f"{i} -> {cls}")

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScaled feature matrix shape:", X_scaled.shape)