from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

# ==================================================
# FLASK APPLICATION
# ==================================================

app = Flask(__name__)
CORS(app)


# ==================================================
# LOAD TRAINED MODEL
# ==================================================

print("==============================================")
print("       AI FRAUD DETECTION API")
print("==============================================")

print("\nLoading trained fraud detection model...")

model = joblib.load("fraud_detection_xgboost.pkl")

preprocessor = joblib.load(
    "fraud_detection_preprocessor.pkl"
)

with open("fraud_threshold.txt", "r") as file:
    threshold = float(file.read())

print("Model loaded successfully!")
print(f"Fraud threshold: {threshold:.2f}")


# ==================================================
# HOME / HEALTH CHECK
# ==================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "success",
        "message": "AI Fraud Detection API is running",
        "model_loaded": True,
        "fraud_threshold": threshold
    })


# ==================================================
# FRAUD ANALYSIS API
# ==================================================

@app.route("/api/analyze", methods=["POST"])
def analyze_transaction():

    try:

        # --------------------------------------------------
        # GET DATA FROM WEBSITE
        # --------------------------------------------------

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No transaction data received"
            }), 400


        transaction_type = str(
            data.get("type", "")
        ).strip().upper()

        amount = float(
            data.get("amount", 0)
        )

        oldbalance_org = float(
            data.get("oldbalanceOrg", 0)
        )

        newbalance_orig = float(
            data.get("newbalanceOrig", 0)
        )

        oldbalance_dest = float(
            data.get("oldbalanceDest", 0)
        )

        newbalance_dest = float(
            data.get("newbalanceDest", 0)
        )

        step = int(
            data.get("step", 0)
        )

        is_flagged_fraud = int(
            data.get("isFlaggedFraud", 0)
        )


        # --------------------------------------------------
        # VALIDATION
        # --------------------------------------------------

        allowed_types = [
            "PAYMENT",
            "TRANSFER",
            "CASH_OUT",
            "DEBIT",
            "CASH_IN"
        ]

        if transaction_type not in allowed_types:

            return jsonify({
                "error": "Invalid transaction type",
                "allowed_types": allowed_types
            }), 400


        if amount < 0:
            return jsonify({
                "error": "Amount cannot be negative"
            }), 400


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
            newbalance_orig == 0
            and oldbalance_org > 0
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

                "oldbalanceOrg":
                    oldbalance_org,

                "newbalanceOrig":
                    newbalance_orig,

                "oldbalanceDest":
                    oldbalance_dest,

                "newbalanceDest":
                    newbalance_dest,

                "isFlaggedFraud":
                    is_flagged_fraud,

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
        # PREPROCESS
        # ==================================================

        processed_transaction = (
            preprocessor.transform(transaction)
        )


        # ==================================================
        # PREDICTION
        # ==================================================

        fraud_probability = float(
    model.predict_proba(
        processed_transaction
    )[0][1]
)


        # ==================================================
        # DETERMINE RISK
        # ==================================================

        is_fraud = (
            fraud_probability >= threshold
        )


        if fraud_probability < threshold:

            risk_level = "LOW"
            risk_status = "GREEN"

        else:

            risk_level = "HIGH"
            risk_status = "RED"


        # Medium-risk warning

        if (
            fraud_probability >= threshold * 0.75
            and
            fraud_probability < threshold
        ):

            risk_level = "MEDIUM"
            risk_status = "YELLOW"


        # ==================================================
        # RISK SIGNALS
        # ==================================================

        risk_signals = []


        if amount > 100000:

            risk_signals.append(
                "Large transaction amount"
            )


        if amount_to_sender_balance > 0.5:

            risk_signals.append(
                "Large amount compared with sender balance"
            )


        if sender_balance_depleted:

            risk_signals.append(
                "Sender balance completely depleted"
            )


        if receiver_initially_zero:

            risk_signals.append(
                "Receiver had zero balance before transaction"
            )


        if sender_balance_error > 1:

            risk_signals.append(
                "Sender balance movement differs from amount"
            )


        if receiver_balance_error > 1:

            risk_signals.append(
                "Receiver balance movement differs from amount"
            )


        if is_flagged_fraud:

            risk_signals.append(
                "Transaction was previously flagged"
            )


        if len(risk_signals) == 0:

            risk_signals.append(
                "No major rule-based warning detected"
            )


        # ==================================================
        # RECOMMENDATION
        # ==================================================

        if is_fraud:

            recommendation = (
                "Send transaction for review."
            )

            classification = "FRAUD"

        else:

            recommendation = (
                "Transaction can proceed."
            )

            classification = "GENUINE"


        # ==================================================
        # API RESPONSE
        # ==================================================

        result = {

            "success": True,

            "transaction_type":
                transaction_type,

            "amount":
                amount,

            "fraud_probability":
                round(
                    fraud_probability * 100,
                    2
                ),

            "risk_level":
                risk_level,

            "risk_status":
                risk_status,

            "classification":
                classification,

            "risk_signals":
                risk_signals,

            "recommendation":
                recommendation
        }


        return jsonify(result)


    # ==================================================
    # ERROR HANDLING
    # ==================================================

    except Exception as e:

        print("ERROR:", str(e))

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ==================================================
# START SERVER
# ==================================================

if __name__ == "__main__":

    print("\n==============================================")
    print("AI FRAUD DETECTION API STARTED")
    print("==============================================")

    print("\nAPI URL:")
    print("http://127.0.0.1:5000")

    print("\nAnalyze endpoint:")
    print("POST http://127.0.0.1:5000/api/analyze")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )