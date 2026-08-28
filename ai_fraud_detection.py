from datasets import load_dataset, concatenate_datasets
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


print("==============================================")
print("        AI FRAUD DETECTION SYSTEM")
print("        STEP 6 - IMPROVED MODEL")
print("==============================================")


# ==================================================
# STEP 1 - LOAD DATASET
# ==================================================

print("\n[1/8] Loading dataset...")

dataset = load_dataset(
    "CiferAI/Cifer-Fraud-Detection-Dataset-AF",
    split="train"
)

print(f"Total transactions available: {len(dataset):,}")


# ==================================================
# STEP 2 - SELECT FRAUD AND GENUINE TRANSACTIONS
# ==================================================

print("\n[2/8] Selecting transactions...")

fraud_data = dataset.filter(
    lambda x: x["isFraud"] == 1
)

genuine_data = dataset.filter(
    lambda x: x["isFraud"] == 0
)

print(f"Available fraud transactions   : {len(fraud_data):,}")
print(f"Available genuine transactions : {len(genuine_data):,}")


# ==================================================
# STEP 3 - CREATE DEVELOPMENT DATASET
# ==================================================

print("\n[3/8] Creating development dataset...")

FRAUD_COUNT = min(10000, len(fraud_data))
GENUINE_COUNT = min(30000, len(genuine_data))

fraud_sample = fraud_data.shuffle(
    seed=42
).select(
    range(FRAUD_COUNT)
)

genuine_sample = genuine_data.shuffle(
    seed=42
).select(
    range(GENUINE_COUNT)
)

working_dataset = concatenate_datasets(
    [fraud_sample, genuine_sample]
)

working_dataset = working_dataset.shuffle(
    seed=42
)

print(f"Fraud transactions   : {FRAUD_COUNT:,}")
print(f"Genuine transactions : {GENUINE_COUNT:,}")
print(f"Total                 : {len(working_dataset):,}")


# ==================================================
# STEP 4 - CONVERT TO PANDAS
# ==================================================

df = working_dataset.to_pandas()


# ==================================================
# STEP 5 - FEATURE ENGINEERING
# ==================================================

print("\n[4/8] Creating fraud detection features...")


# How much money left the sender
df["sender_balance_change"] = (
    df["oldbalanceOrg"] - df["newbalanceOrig"]
)


# How much money reached the receiver
df["receiver_balance_change"] = (
    df["newbalanceDest"] - df["oldbalanceDest"]
)


# Transaction amount compared with sender balance
df["amount_to_sender_balance"] = (
    df["amount"] /
    (df["oldbalanceOrg"] + 1)
)


# Difference between transaction amount
# and actual sender balance movement
df["sender_balance_error"] = (
    abs(
        df["amount"] -
        df["sender_balance_change"]
    )
)


# Difference between transaction amount
# and actual receiver balance movement
df["receiver_balance_error"] = (
    abs(
        df["amount"] -
        df["receiver_balance_change"]
    )
)


# Whether sender balance became zero
df["sender_balance_depleted"] = (
    (df["newbalanceOrig"] == 0) &
    (df["oldbalanceOrg"] > 0)
).astype(int)


# Whether receiver initially had zero balance
df["receiver_initially_zero"] = (
    df["oldbalanceDest"] == 0
).astype(int)


# Large transaction indicator
df["large_transaction"] = (
    df["amount"] > 100000
).astype(int)


# Percentage of sender balance transferred
df["sender_transfer_ratio"] = (
    df["amount"] /
    (df["oldbalanceOrg"] + 1)
)


# ==================================================
# STEP 6 - SELECT FEATURES
# ==================================================

features = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",

    "sender_balance_change",
    "receiver_balance_change",
    "amount_to_sender_balance",
    "sender_balance_error",
    "receiver_balance_error",
    "sender_balance_depleted",
    "receiver_initially_zero",
    "large_transaction",
    "sender_transfer_ratio"
]


X = df[features]
y = df["isFraud"]


print("\nFeatures used by model:")

for feature in features:
    print(" -", feature)


# ==================================================
# STEP 7 - TRAIN / VALIDATION / TEST SPLIT
# ==================================================

print("\n[5/8] Creating train/validation/test split...")


# First split:
# 70% training
# 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# Second split:
# 15% validation
# 15% test
X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


print(f"Training data   : {len(X_train):,}")
print(f"Validation data : {len(X_validation):,}")
print(f"Testing data    : {len(X_test):,}")


# ==================================================
# STEP 8 - ENCODE TRANSACTION TYPE
# ==================================================

categorical_features = [
    "type"
]

numeric_features = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
    "sender_balance_change",
    "receiver_balance_change",
    "amount_to_sender_balance",
    "sender_balance_error",
    "receiver_balance_error",
    "sender_balance_depleted",
    "receiver_initially_zero",
    "large_transaction",
    "sender_transfer_ratio"
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "type",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ==================================================
# TRAIN XGBOOST
# ==================================================

print("\n[6/8] Training XGBoost fraud detection model...")


model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)


# Transform training data
X_train_processed = preprocessor.fit_transform(
    X_train
)

X_validation_processed = preprocessor.transform(
    X_validation
)

X_test_processed = preprocessor.transform(
    X_test
)


# Train
model.fit(
    X_train_processed,
    y_train
)


print("\nModel training completed successfully!")


# ==================================================
# VALIDATION - FIND A GOOD THRESHOLD
# ==================================================

print("\n[7/8] Finding fraud detection threshold...")


validation_probability = model.predict_proba(
    X_validation_processed
)[:, 1]


best_threshold = 0.50
best_f1 = 0


for threshold in [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80
]:

    validation_prediction = (
        validation_probability >= threshold
    ).astype(int)

    current_f1 = f1_score(
        y_validation,
        validation_prediction
    )

    if current_f1 > best_f1:
        best_f1 = current_f1
        best_threshold = threshold


print(f"Best threshold : {best_threshold:.2f}")
print(f"Validation F1  : {best_f1:.4f}")


# ==================================================
# FINAL TEST
# ==================================================

print("\n[8/8] Evaluating final model...")


test_probability = model.predict_proba(
    X_test_processed
)[:, 1]


test_prediction = (
    test_probability >= best_threshold
).astype(int)


# ==================================================
# CALCULATE METRICS
# ==================================================

precision = precision_score(
    y_test,
    test_prediction,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_prediction,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_prediction,
    zero_division=0
)

accuracy = accuracy_score(
    y_test,
    test_prediction
)


# ==================================================
# DISPLAY PERFORMANCE
# ==================================================

print("\n==============================================")
print("           FINAL MODEL PERFORMANCE")
print("==============================================")

print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"Accuracy  : {accuracy:.4f}")

print(f"\nFraud threshold used: {best_threshold:.2f}")


# ==================================================
# CLASSIFICATION REPORT
# ==================================================

print("\n==============================================")
print("             CLASSIFICATION REPORT")
print("==============================================")

print(
    classification_report(
        y_test,
        test_prediction,
        target_names=[
            "Genuine",
            "Fraud"
        ],
        zero_division=0
    )
)


# ==================================================
# CONFUSION MATRIX
# ==================================================

print("\n==============================================")
print("               CONFUSION MATRIX")
print("==============================================")

matrix = confusion_matrix(
    y_test,
    test_prediction
)

print(matrix)


# ==================================================
# SAVE MODEL
# ==================================================

print("\n==============================================")
print("                SAVING MODEL")
print("==============================================")


joblib.dump(
    model,
    "fraud_detection_xgboost.pkl"
)

joblib.dump(
    preprocessor,
    "fraud_detection_preprocessor.pkl"
)


# Save threshold
with open(
    "fraud_threshold.txt",
    "w"
) as file:

    file.write(
        str(best_threshold)
    )


print("\nFiles created:")
print(" - fraud_detection_xgboost.pkl")
print(" - fraud_detection_preprocessor.pkl")
print(" - fraud_threshold.txt")


print("\n==============================================")
print("       STEP 6 COMPLETED SUCCESSFULLY!")
print("==============================================")