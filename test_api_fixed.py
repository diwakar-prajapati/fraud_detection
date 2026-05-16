"""
Fixed test script for Fraud Detection API
"""

import requests
import json

print("=" * 60)
print("🔍 FRAUD DETECTION API TEST")
print("=" * 60)

# Test health endpoint
print("\n1️⃣ Health Check:")
try:
    response = requests.get("http://localhost:8000/health")
    print(f"   Status: {response.json()}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    exit(1)

# Test simple prediction endpoint (recommended)
print("\n2️⃣ Testing Simple Prediction Endpoint:")
simple_txn = {
    "amount": 2500.00,
    "time_hour": 3,
    "location_id": 99,
    "merchant_id": 999,
    "device_type": "desktop",
    "previous_frauds": 2,
    "days_since_last_txn": 0.5,
    "txn_count_last_24h": 15,
    "avg_amount_last_7d": 50.00
}

try:
    response = requests.post("http://localhost:8000/predict/simple", json=simple_txn)
    result = response.json()
    print(f"   ✅ Transaction ID: {result.get('transaction_id', 'N/A')}")
    print(f"   📊 Fraud Probability: {result.get('fraud_probability', 'N/A')}%")
    print(f"   ⚠️ Risk Level: {result.get('fraud_risk_level', 'N/A')}")
    print(f"   💡 Recommendation: {result.get('recommendation', 'N/A')}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test multiple scenarios
print("\n3️⃣ Testing Multiple Scenarios:")
print("-" * 40)

test_cases = [
    {"name": "Normal Transaction", "amount": 25.50, "time_hour": 14, "previous_frauds": 0, "txn_count_last_24h": 1},
    {"name": "Suspicious High Amount", "amount": 3500.00, "time_hour": 3, "previous_frauds": 0,
     "txn_count_last_24h": 1},
    {"name": "Late Night Multiple", "amount": 150.00, "time_hour": 2, "previous_frauds": 0, "txn_count_last_24h": 12},
    {"name": "Fraud History", "amount": 500.00, "time_hour": 15, "previous_frauds": 3, "txn_count_last_24h": 2},
]

for test in test_cases:
    txn = {
        "amount": test["amount"],
        "time_hour": test["time_hour"],
        "previous_frauds": test["previous_frauds"],
        "txn_count_last_24h": test["txn_count_last_24h"],
        "location_id": 1,
        "merchant_id": 100,
        "device_type": "mobile",
        "days_since_last_txn": 30,
        "avg_amount_last_7d": 50.00
    }

    try:
        response = requests.post("http://localhost:8000/predict/simple", json=txn)
        result = response.json()
        print(f"\n   📍 {test['name']}:")
        print(f"      Amount: ${test['amount']}, Hour: {test['time_hour']}h")
        print(f"      → {result.get('fraud_probability', 'N/A')}% - {result.get('fraud_risk_level', 'N/A')}")
    except Exception as e:
        print(f"   ❌ Error for {test['name']}: {e}")

print("\n" + "=" * 60)
print("✅ TEST COMPLETE")
print("=" * 60)
print("\n💡 Note: Using rule-based detection for demo.")
print("   For full ML model, ensure model is properly trained and features match.")