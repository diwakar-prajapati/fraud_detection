"""
Complete Training Pipeline for ALL Models
Run: python main.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

# Import all theory modules
from src.statistical_learning import StatisticalLearning
from src.ml_algorithms import MLAlgorithms
from src.deep_learning import DeepLearningTheory


def main():
    print("=" * 70)
    print("🔬 COMPLETE ML/DL FRAUD DETECTION SYSTEM")
    print("=" * 70)

    # Step 1: Load data
    print("\n📥 Loading datasets...")
    df = load_data()

    # Step 2: Statistical Learning Theory
    print("\n" + "=" * 70)
    print("📊 PHASE 1: STATISTICAL LEARNING THEORY")
    print("=" * 70)
    stats = StatisticalLearning(df)
    stats.all_statistical_concepts()

    # Step 3: ML Algorithms Theory
    print("\n" + "=" * 70)
    print("🤖 PHASE 2: MACHINE LEARNING ALGORITHMS")
    print("=" * 70)
    ml = MLAlgorithms()
    ml.all_supervised_algorithms()
    ml.unsupervised_learning()

    # Step 4: Deep Learning Theory
    print("\n" + "=" * 70)
    print("🧠 PHASE 3: DEEP LEARNING THEORY")
    print("=" * 70)
    dl = DeepLearningTheory(input_dim=28)
    dl.all_deep_learning_concepts()

    # Step 5: Train actual models
    print("\n" + "=" * 70)
    print("🏋️ PHASE 4: MODEL TRAINING")
    print("=" * 70)

    X_train, X_test, y_train, y_test, scaler = prepare_data(df)

    # Train Random Forest
    print("\n🌲 Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_score = rf.score(X_test, y_test)
    print(f"   RF Accuracy: {rf_score:.4f}")

    # Train XGBoost
    print("\n⚡ Training XGBoost...")
    xgb = XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=5)
    xgb.fit(X_train, y_train)
    xgb_score = xgb.score(X_test, y_test)
    print(f"   XGB Accuracy: {xgb_score:.4f}")

    # Train Deep Learning
    print("\n🧠 Training Neural Network...")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

    nn_model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    nn_model.fit(X_train, y_train, epochs=10, batch_size=256, validation_split=0.2, verbose=0)
    nn_score = nn_model.evaluate(X_test, y_test, verbose=0)[1]
    print(f"   NN Accuracy: {nn_score:.4f}")

    # Step 6: Save models
    print("\n💾 Saving models...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(xgb, 'models/best_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(list(X_train.columns), 'models/feature_names.pkl')
    nn_model.save('models/nn_model.h5')

    print("\n" + "=" * 70)
    print("✅ COMPLETE! All models trained and saved")
    print("=" * 70)

    print("\n📊 FINAL RESULTS:")
    print(f"   Random Forest: {rf_score * 100:.2f}% accuracy")
    print(f"   XGBoost: {xgb_score * 100:.2f}% accuracy")
    print(f"   Neural Network: {nn_score * 100:.2f}% accuracy")

    print("\n🚀 NEXT STEPS:")
    print("   1. Run API: uvicorn application.bank_api:app --reload")
    print("   2. Run Dashboard: streamlit run application/dashboard.py")
    print("   3. Deploy to Render: python scripts/deploy_render.py")

    return xgb, rf, nn_model


def load_data():
    """Load the best available dataset"""
    if os.path.exists('data/creditcard.csv'):
        df = pd.read_csv('data/creditcard.csv')
        print(f"   ✅ Loaded creditcard.csv: {len(df)} transactions")
    elif os.path.exists('data/synthetic_fraud.csv'):
        df = pd.read_csv('data/synthetic_fraud.csv')
        print(f"   ✅ Loaded synthetic_fraud.csv: {len(df)} transactions")
    else:
        print("   ⚠️ No dataset found. Run download script first.")
        return None

    print(
        f"   Fraud rate: {df['Class'].mean() * 100:.4f}%" if 'Class' in df.columns else f"   Fraud rate: {df['class'].mean() * 100:.4f}%")
    return df


def prepare_data(df):
    """Prepare data for training"""
    # Handle column names
    if 'Class' in df.columns:
        y = df['Class']
        X = df.drop(['Class', 'Time'], axis=1) if 'Time' in df.columns else df.drop('Class', axis=1)
    else:
        y = df['class']
        X = df.drop('class', axis=1)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    print(f"\n📊 Data prepared:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing: {len(X_test)} samples")
    print(f"   Features: {len(X.columns)}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == "__main__":
    main()