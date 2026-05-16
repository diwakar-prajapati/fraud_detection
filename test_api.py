import requests
import json

print("="*60)
print("🔍 FRAUD DETECTION API TEST")
print("="*60)

# Test 1: Health Check
print("\n1️⃣ Health Check:")
response = requests.get("http://localhost:8000/health")
print(f"   Status: {response.json()}")

# Test 2: Normal Transaction
print("\n2️⃣ Normal Transaction Test:")
normal = {
    "amount": 25.50,
    "time_hour": 14,
    "location_id": 1,
    "merchant_id": 100,
    "device_type": "mobile",
    "previous_frauds": 0,
    "days_since_last_txn": 30,
    "txn_count_last_24h": 1,
    "avg_amount_last_7d": 35.00
}
response = requests.post("http://localhost:8000/predict", json=normal)
result = response.json()
print(f"   Fraud Probability: {result['fraud_probability']}%")
print(f"   Risk Level: {result['fraud_risk_level']}")
print(f"   Recommendation: {result['recommendation']}")

# Test 3: Suspicious Transaction
print("\n3️⃣ Suspicious Transaction Test:")
suspicious = {
    "amount": 3500.00,
    "time_hour": 3,
    "location_id": 99,
    "merchant_id": 999,
    "device_type": "desktop",
    "previous_frauds": 3,
    "days_since_last_txn": 0.1,
    "txn_count_last_24h": 25,
    "avg_amount_last_7d": 50.00
}
response = requests.post("http://localhost:8000/predict", json=suspicious)
result = response.json()
print(f"   Fraud Probability: {result['fraud_probability']}%")
print(f"   Risk Level: {result['fraud_risk_level']}")
print(f"   Recommendation: {result['recommendation']}")

# Test 4: Edge Case - High Amount Day Time
print("\n4️⃣ High Amount Day Time Test:")
high_amount_day = {
    "amount": 5000.00,
    "time_hour": 14,
    "location_id": 1,
    "merchant_id": 100,
    "device_type": "mobile",
    "previous_frauds": 0,
    "days_since_last_txn": 1,
    "txn_count_last_24h": 1,
    "avg_amount_last_7d": 100.00
}
response = requests.post("http://localhost:8000/predict", json=high_amount_day)
result = response.json()
print(f"   Fraud Probability: {result['fraud_probability']}%")
print(f"   Risk Level: {result['fraud_risk_level']}")

print("\n" + "="*60)
print("✅ API TEST COMPLETE")
print("="*60)