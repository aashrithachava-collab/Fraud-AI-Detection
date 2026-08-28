import os
import razorpay
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Get Razorpay Test Mode credentials
KEY_ID = os.getenv("RAZORPAY_KEY_ID")
KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")

print("==============================================")
print("          RAZORPAY TEST MODE")
print("==============================================")

# Check credentials
if not KEY_ID or not KEY_SECRET:
    print("ERROR: Razorpay credentials not found.")
    print("Check your .env file.")
    exit()

if not KEY_ID.startswith("rzp_test_"):
    print("ERROR: This is not a Razorpay Test Mode key.")
    exit()

print("Test API key detected successfully.")

# Create Razorpay client
client = razorpay.Client(
    auth=(KEY_ID, KEY_SECRET)
)

print("Connected to Razorpay successfully!")

# Create a test order
order_data = {
    "amount": 50000,       # ₹500
    "currency": "INR",
    "receipt": "fraud_test_001",
    "notes": {
        "project": "AI Fraud Detection",
        "purpose": "Razorpay Buildathon testing"
    }
}

try:

    order = client.order.create(data=order_data)

    print("\n==============================================")
    print("          TEST ORDER CREATED")
    print("==============================================")

    print("Order ID :", order["id"])
    print("Amount   : ₹", order["amount"] / 100)
    print("Currency :", order["currency"])
    print("Status   :", order["status"])

    print("\n==============================================")
    print("       RAZORPAY TEST SUCCESSFUL")
    print("==============================================")

except Exception as e:

    print("\n==============================================")
    print("          RAZORPAY TEST FAILED")
    print("==============================================")

    print("Error:", e)