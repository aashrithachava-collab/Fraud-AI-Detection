import joblib
import pandas as pd


print("==============================================")
print("          AI FRAUD DETECTION")
print("          REAL-TIME TRANSACTION CHECK")
print("==============================================")


# ==================================================
# LOAD TRAINED MODEL
# ==================================================

print("\nLoading trained fraud detection model...")

model = joblib.load(
    "fraud_detection_xgboost.pkl"
)

preprocessor = joblib.load(
    "fraud_detection_preprocessor.pkl"
)

with open("fraud_threshold.txt", "r") as file:
    threshold = float(file.read())


print("Model loaded successfully!")
print(f"Fraud threshold: {threshold:.2f}")


# ==================================================
# GET TRANSACTION DETAILS
# ==================================================

print("\n==============================================")
print("ENTER TRANSACTION DETAILS")
print("==============================================")


transaction_type = input(
    "Transaction type (PAYMENT / TRANSFER / CASH_OUT / DEBIT / CASH_IN): "
).strip().upper()

amount = float(
    input("Transaction amount: ")
)

oldbalance_org = float(
    input("Sender balance BEFORE transaction: ")
)

newbalance_orig = float(
    input("Sender balance AFTER transaction: ")
)

oldbalance_dest = float(
    input("Receiver balance BEFORE transaction: ")
)

newbalance_dest = float(
    input("Receiver balance AFTER transaction: ")
)

step = int(
    input("Transaction step/time number: ")
)

is_flagged_fraud = int(
    input("Already flagged by system? (0 = No, 1 = Yes): ")
)


# ==================================================
# CREATE FEATURES
# ==================================================

sender_balance_change = (
    oldbalance_org - newbalance_orig
)

receiver_balance_change = (
    newbalance_dest - oldbalance_dest
)

amount_to_sender_balance = (
    amount / (oldbalance_org + 1)
)

sender_balance_error = abs(
    amount - sender_balance_change
)

receiver_balance_error = abs(
    amount - receiver_balance_change
)

sender_balance_depleted = int(
    newbalance_orig == 0 and oldbalance_org > 0
)

receiver_initially_zero = int(
    oldbalance_dest == 0
)

large_transaction = int(
    amount > 100000
)

sender_transfer_ratio = (
    amount / (oldbalance_org + 1)
)


# ==================================================
# CREATE TRANSACTION DATAFRAME
# ==================================================

transaction = pd.DataFrame([
    {
        "step": step,
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalance_org,
        "newbalanceOrig": newbalance_orig,
        "oldbalanceDest": oldbalance_dest,
        "newbalanceDest": newbalance_dest,
        "isFlaggedFraud": is_flagged_fraud,

        "sender_balance_change":
            sender_balance_change,

        "receiver_balance_change":
            receiver_balance_change,

        "amount_to_sender_balance":
            amount_to_sender_balance,

        "sender_balance_error":
            sender_balance_error,

        "receiver_balance_error":
            receiver_balance_error,

        "sender_balance_depleted":
            sender_balance_depleted,

        "receiver_initially_zero":
            receiver_initially_zero,

        "large_transaction":
            large_transaction,

        "sender_transfer_ratio":
            sender_transfer_ratio
    }
])


# ==================================================
# PREPROCESS TRANSACTION
# ==================================================

processed_transaction = preprocessor.transform(
    transaction
)


# ==================================================
# GET FRAUD PROBABILITY
# ==================================================

fraud_probability = model.predict_proba(
    processed_transaction
)[0][1]


# ==================================================
# DETERMINE RISK
# ==================================================

is_fraud = (
    fraud_probability >= threshold
)

# ==================================================
# DETERMINE RISK LEVEL
# ==================================================

if fraud_probability < threshold:

    risk_level = "LOW"
    risk_symbol = "GREEN"

else:

    risk_level = "HIGH"
    risk_symbol = "RED"


# Additional warning level for transactions
# approaching the fraud threshold

if fraud_probability >= threshold * 0.75 and fraud_probability < threshold:

    risk_level = "MEDIUM"
    risk_symbol = "YELLOW"


# Final fraud decision

is_fraud = (
    fraud_probability >= threshold
)


# ==================================================
# DISPLAY RESULT
# ==================================================

print("\n")
print("==============================================")
print("             FRAUD ANALYSIS RESULT")
print("==============================================")

print(f"\nTransaction type : {transaction_type}")
print(f"Amount           : ₹{amount:,.2f}")

print(
    f"\nFraud probability: "
    f"{fraud_probability * 100:.2f}%"
)

print(f"Risk level       : {risk_level}")
print(f"Risk status      : {risk_symbol}")


if is_fraud:

    print("\n🔴 POSSIBLE FRAUD DETECTED")
    print("Recommendation: Send transaction for review.")

else:

    print("\n🟢 TRANSACTION APPEARS GENUINE")
    print("Recommendation: Transaction can proceed.")


# ==================================================
# SHOW RISK SIGNALS
# ==================================================

print("\n==============================================")
print("             RISK SIGNALS")
print("==============================================")


if amount > 100000:

    print("⚠ Large transaction amount")


if amount_to_sender_balance > 0.5:

    print("⚠ Large amount compared with sender balance")


if sender_balance_depleted:

    print("⚠ Sender balance completely depleted")


if receiver_initially_zero:

    print("⚠ Receiver had zero balance before transaction")


if sender_balance_error > 1:

    print("⚠ Sender balance movement differs from amount")


if receiver_balance_error > 1:

    print("⚠ Receiver balance movement differs from amount")


if is_flagged_fraud:

    print("⚠ Transaction was previously flagged")


if (
    amount <= 100000
    and
    not sender_balance_depleted
    and
    not receiver_initially_zero
    and
    sender_balance_error <= 1
    and
    receiver_balance_error <= 1
):

    print("✓ No major rule-based warning detected")


print("\n==============================================")
print("           ANALYSIS COMPLETED")
print("==============================================")