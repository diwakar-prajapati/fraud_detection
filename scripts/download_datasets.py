"""
Download ALL free datasets for fraud detection
Run this first: python scripts/download_datasets.py
"""

import os
import urllib.request
import pandas as pd
import numpy as np
from zipfile import ZipFile
import io


def download_datasets():
    """Download all free datasets"""

    os.makedirs('data', exist_ok=True)

    print("=" * 60)
    print("📥 DOWNLOADING FRAUD DETECTION DATASETS")
    print("=" * 60)

    # Dataset 1: Credit Card Fraud (Kaggle - most popular)
    print("\n1️⃣ Downloading Credit Card Fraud Dataset...")
    url1 = "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"
    try:
        urllib.request.urlretrieve(url1, "data/creditcard.csv")
        print("   ✅ creditcard.csv downloaded (284,807 transactions)")
    except:
        print("   ⚠️ Direct download failed, generating synthetic data...")
        generate_synthetic_creditcard()

    # Dataset 2: Alternative CC Fraud (smaller, for quick testing)
    print("\n2️⃣ Downloading Alternative Fraud Dataset...")
    url2 = "https://raw.githubusercontent.com/curiousily/Credit-Card-Fraud-Detection/master/data/creditcard.csv"
    try:
        urllib.request.urlretrieve(url2, "data/creditcard_alt.csv")
        print("   ✅ creditcard_alt.csv downloaded")
    except:
        print("   ⚠️ Using primary dataset as fallback")

    # Dataset 3: Generate synthetic dataset with known patterns
    print("\n3️⃣ Generating Synthetic Fraud Dataset...")
    generate_synthetic_fraud()

    print("\n" + "=" * 60)
    print("✅ ALL DATASETS READY!")
    print("=" * 60)

    # Verify downloads
    for file in os.listdir('data'):
        if file.endswith('.csv'):
            size = os.path.getsize(f'data/{file}') / (1024 * 1024)
            print(f"   📊 {file}: {size:.2f} MB")


def generate_synthetic_creditcard():
    """Generate synthetic credit card data"""
    np.random.seed(42)
    n_transactions = 100000
    n_features = 28

    # Generate features
    X = np.random.randn(n_transactions, n_features)

    # Add some correlation
    for i in range(1, n_features):
        X[:, i] += 0.3 * X[:, i - 1]

    # Generate fraud labels (0.5% fraud rate)
    fraud_proba = 1 / (1 + np.exp(-(X[:, 0] * 2 + X[:, 1] * 1.5)))
    y = (fraud_proba > 0.99).astype(int)

    # Add amount
    amount = np.random.exponential(100, n_transactions)
    amount[y == 1] = np.random.exponential(500, y.sum())

    # Create dataframe
    df = pd.DataFrame(X, columns=[f'V{i + 1}' for i in range(n_features)])
    df['Amount'] = amount
    df['Class'] = y
    df['Time'] = np.arange(n_transactions)

    df.to_csv('data/creditcard.csv', index=False)
    print(f"   ✅ Generated {n_transactions} synthetic transactions ({y.sum()} frauds)")


def generate_synthetic_fraud():
    """Generate synthetic dataset with clear patterns"""
    np.random.seed(42)

    n_normal = 50000
    n_fraud = 500

    # Normal transactions
    normal_amount = np.random.exponential(80, n_normal)
    normal_location = np.random.choice([1, 2, 3, 4, 5], n_normal, p=[0.6, 0.2, 0.1, 0.05, 0.05])
    normal_time = np.random.uniform(0, 24, n_normal)

    # Fraud transactions (different patterns)
    fraud_amount = np.random.exponential(500, n_fraud)
    fraud_location = np.random.choice([1, 2, 3, 4, 5], n_fraud, p=[0.1, 0.1, 0.2, 0.3, 0.3])
    fraud_time = np.random.uniform(0, 5, n_fraud)  # Late night frauds

    # Combine
    df = pd.DataFrame({
        'amount': np.concatenate([normal_amount, fraud_amount]),
        'location_id': np.concatenate([normal_location, fraud_location]),
        'hour': np.concatenate([normal_time, fraud_time]),
        'days_since_last_txn': np.random.exponential(30, n_normal + n_fraud),
        'txn_count_7d': np.random.poisson(5, n_normal + n_fraud),
        'class': np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])
    })

    # Add engineered features
    df['amount_log'] = np.log1p(df['amount'])
    df['night_txn'] = (df['hour'] < 6).astype(int)
    df['high_amount'] = (df['amount'] > 200).astype(int)

    df.to_csv('data/synthetic_fraud.csv', index=False)
    print(f"   ✅ Generated synthetic_fraud.csv ({n_fraud} frauds, {n_normal} normal)")


if __name__ == "__main__":
    download_datasets()