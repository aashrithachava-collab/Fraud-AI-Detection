import requests

url = "http://127.0.0.1:5000/api/analyze"

transaction = {
    "type": "CASH_OUT",
    "amount": 365456,
    "oldbalanceOrg": 451942,
    "newbalanceOrig": 86500,
    "oldbalanceDest": 465964,
    "newbalanceDest": 903994,
    "step": 295,
    "isFlaggedFraud": 0
}

response = requests.post(
    url,
    json=transaction
)

print("\n================================")
print("       API RESPONSE")
print("================================")

print(response.json())