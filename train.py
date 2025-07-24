import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import pickle
import seaborn as sns

# Ensure 'app/data' folder exists
os.makedirs("app/data", exist_ok=True)

# Load the dataset
df = sns.load_dataset("penguins")
df = df.dropna()

# Drop 'year' column only if it exists
if 'year' in df.columns:
    df = df.drop(columns=['year'])

# Separate features and target
X = df.drop("species", axis=1)
y = df["species"]

# One-hot encode categorical columns
X = pd.get_dummies(X, columns=["sex", "island"])

# Encode the target labels (species) to numeric values
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save the label encoder for decoding predictions later
with open("app/data/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Convert to DMatrix (XGBoost's data structure)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Define parameters
params = {
    "objective": "multi:softmax",  # output the predicted class directly
    "num_class": 3,
    "eval_metric": "mlogloss",
    "seed": 42
}

# Train the model
model = xgb.train(params, dtrain, num_boost_round=50)

# Predict
train_preds = model.predict(dtrain)
test_preds = model.predict(dtest)

# Evaluate with F1 score
print("Train F1:", f1_score(y_train, train_preds, average="macro"))
print("Test F1:", f1_score(y_test, test_preds, average="macro"))

# Save the model using XGBoost's save_model method inside 'app/data' folder
model.save_model("app/data/xgb_penguin_model.json")

print("✅ Model and label encoder saved successfully inside 'app/data' folder!")
