"""
THEORY MODULE 2: All Machine Learning Algorithms
Covers: Supervised, Unsupervised, Ensemble, Evaluation
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt


class MLAlgorithms:
    """Complete ML algorithms implementation with theory"""

    def __init__(self):
        self.models = {}
        self.results = {}

    def all_supervised_algorithms(self):
        """Implement ALL supervised learning algorithms"""

        print("\n" + "=" * 70)
        print("🤖 SUPERVISED LEARNING ALGORITHMS")
        print("=" * 70)

        algorithms = {
            'Logistic Regression': self._logistic_regression_theory,
            'Decision Tree': self._decision_tree_theory,
            'Random Forest': self._random_forest_theory,
            'XGBoost': self._xgboost_theory,
            'SVM': self._svm_theory,
            'KNN': self._knn_theory,
            'Naive Bayes': self._naive_bayes_theory,
            'Gradient Boosting': self._gradient_boosting_theory
        }

        for name, theory_func in algorithms.items():
            theory_func()

    def _logistic_regression_theory(self):
        """Logistic Regression theory"""
        print("\n" + "=" * 60)
        print("📐 LOGISTIC REGRESSION")
        print("=" * 60)

        print("\n🎯 WHAT IT DOES:")
        print("   Predicts probability of binary outcome (fraud vs normal)")

        print("\n📐 MATHEMATICAL FORMULA:")
        print("   P(y=1|x) = 1 / (1 + e^-(β₀ + β₁x₁ + β₂x₂ + ...))")
        print("   where:")
        print("     - P(y=1|x): Probability of fraud")
        print("     - β: Coefficients (learned from data)")
        print("     - x: Features (amount, location, time, etc.)")

        print("\n🔍 HOW IT WORKS:")
        print("   1. Linear combination: z = β₀ + β₁x₁ + ...")
        print("   2. Sigmoid function: σ(z) = 1/(1+e^(-z))")
        print("   3. Output between 0 and 1 (probability)")
        print("   4. Threshold at 0.5 for classification")

        print("\n💡 WHEN TO USE:")
        print("   ✅ Binary classification")
        print("   ✅ When you need probability scores")
        print("   ✅ When features are linearly separable")
        print("   ❌ Complex non-linear relationships")

        print("\n📊 INTERPRETATION:")
        print("   Coefficients tell you feature importance")
        print("   Positive coefficient → increases fraud probability")
        print("   Negative coefficient → decreases fraud probability")

    def _decision_tree_theory(self):
        """Decision Tree theory"""
        print("\n" + "=" * 60)
        print("🌳 DECISION TREE")
        print("=" * 60)

        print("\n🎯 WHAT IT DOES:")
        print("   Splits data based on feature values to make decisions")

        print("\n📐 SPLITTING CRITERIA:")
        print("   1. Gini Impurity: 1 - Σ(pᵢ²)")
        print("      - Measures how 'pure' a node is")
        print("      - 0 = pure (all same class), 0.5 = impure")

        print("   2. Entropy: -Σ(pᵢ * log₂(pᵢ))")
        print("      - Measures uncertainty/information")
        print("      - Used in ID3, C4.5 algorithms")

        print("   3. Information Gain: Entropy(parent) - Σ(weighted entropy(child))")
        print("      - Measures how much a split reduces uncertainty")

        print("\n🔍 EXAMPLE - Fraud Detection:")
        print("   Root: All transactions (50% fraud, 50% normal)")
        print("   ├── Amount > $500?")
        print("   │   ├── Yes: 80% fraud, 20% normal → High risk")
        print("   │   └── No: 20% fraud, 80% normal → Low risk")
        print("   │       ├── Night transaction?")
        print("   │       │   ├── Yes: 40% fraud → Medium risk")
        print("   │       │   └── No: 10% fraud → Low risk")

        print("\n💡 ADVANTAGES:")
        print("   ✅ Easy to understand and explain")
        print("   ✅ Handles numerical and categorical data")
        print("   ✅ No feature scaling needed")
        print("   ✅ Shows the decision path")

        print("\n⚠️ DISADVANTAGES:")
        print("   ❌ Prone to overfitting (use max_depth)")
        print("   ❌ Unstable (small changes = different tree)")

    def _random_forest_theory(self):
        """Random Forest theory - Ensemble method"""
        print("\n" + "=" * 60)
        print("🌲 RANDOM FOREST (Ensemble Method)")
        print("=" * 60)

        print("\n🎯 WHAT IT DOES:")
        print("   Combines multiple decision trees for better accuracy")

        print("\n🔧 HOW IT WORKS (Bagging):")
        print("   1. Bootstrap Sampling:")
        print("      - Create N subsets of data (sampling with replacement)")
        print("      - Each subset has ~63% of original data")
        print("      - Remaining 37% is 'out-of-bag' for validation")

        print("\n   2. Feature Randomization:")
        print("      - At each split, consider √n random features")
        print("      - Reduces correlation between trees")

        print("\n   3. Voting:")
        print("      - Each tree votes on fraud/normal")
        print("      - Majority wins (classification)")
        print("      - Average for regression")

        print("\n📊 WHY IT WORKS:")
        print("   🎯 Low Bias: Each tree can fit complex patterns")
        print("   🎯 Low Variance: Averaging reduces overfitting")
        print("   🎯 Error = Bias² + Variance + Irreducible Error")

        print("\n🎮 HYPERPARAMETERS:")
        print("   n_estimators: Number of trees (more = better, but slower)")
        print("   max_depth: Tree depth (limits overfitting)")
        print("   min_samples_split: Minimum samples to split node")
        print("   max_features: Features to consider per split")

    def _xgboost_theory(self):
        """XGBoost theory - Gradient Boosting"""
        print("\n" + "=" * 60)
        print("⚡ XGBOOST (Extreme Gradient Boosting)")
        print("=" * 60)

        print("\n🎯 WHAT IT DOES:")
        print("   Sequentially builds trees that correct previous errors")

        print("\n🔧 HOW IT WORKS (Boosting):")
        print("   Tree 1: Initial prediction")
        print("   Tree 2: Learns from errors of Tree 1")
        print("   Tree 3: Learns from errors of Tree 1+2")
        print("   ... and so on")

        print("\n📐 MATHEMATICAL FORMULA:")
        print("   F(x) = Σ fₜ(x) for t = 1 to T")
        print("   where fₜ(x) is tree that minimizes:")
        print("     Loss = Σ L(y, ŷ) + Σ Ω(fₜ)")
        print("     Ω(f) = γT + ½λ||w||² (regularization)")

        print("\n🎯 WHY IT'S BEST FOR FRAUD DETECTION:")
        print("   ✅ Handles imbalanced data (scale_pos_weight)")
        print("   ✅ Built-in regularization (prevents overfitting)")
        print("   ✅ Missing value handling")
        print("   ✅ Feature importance built-in")
        print("   ✅ Very fast training")

        print("\n⚙️ KEY PARAMETERS:")
        print("   learning_rate: Step size (0.01-0.3)")
        print("   max_depth: Tree depth (3-10)")
        print("   subsample: Row sampling (0.5-1)")
        print("   colsample_bytree: Column sampling")
        print("   scale_pos_weight: For imbalanced data")

    def _svm_theory(self):
        """Support Vector Machine theory"""
        print("\n" + "=" * 60)
        print("🎯 SUPPORT VECTOR MACHINE (SVM)")
        print("=" * 60)

        print("\n🎯 WHAT IT DOES:")
        print("   Finds the optimal hyperplane that separates classes")

        print("\n📐 MATHEMATICAL FORMULA:")
        print("   Maximize margin = 2/||w||")
        print("   Subject to: yᵢ(w·xᵢ + b) ≥ 1")

        print("\n🔍 KERNEL TRICK:")
        print("   Maps data to higher dimensions for non-linear separation")
        print("   Kernels:")
        print("     - Linear: K(x,y) = x·y")
        print("     - Polynomial: K(x,y) = (x·y + c)ᵈ")
        print("     - RBF: K(x,y) = exp(-γ||x-y||²)")
        print("     - Sigmoid: K(x,y) = tanh(αx·y + c)")

        print("\n💡 FOR FRAUD DETECTION:")
        print("   ✅ Good for high-dimensional data")
        print("   ✅ Memory efficient (uses support vectors only)")
        print("   ❌ Slow for large datasets")
        print("   ❌ Hard to interpret")

    def _knn_theory(self):
        """K-Nearest Neighbors theory"""
        print("\n" + "=" * 60)
        print("👥 K-NEAREST NEIGHBORS (KNN)")
        print("=" * 60)

        print("\n🎯 WHAT IT DOES:")
        print("   Classifies based on majority vote of K closest neighbors")

        print("\n📐 DISTANCE METRICS:")
        print("   1. Euclidean: √Σ(xᵢ - yᵢ)²")
        print("   2. Manhattan: Σ|xᵢ - yᵢ|")
        print("   3. Minkowski: (Σ|xᵢ - yᵢ|ᵖ)^(1/p)")

        print("\n🔍 EXAMPLE:")
        print("   K=3, Transaction: Amount=$500, Hour=2AM")
        print("   Neighbors:")
        print("     - Txn1: Fraud (distance=5)")
        print("     - Txn2: Fraud (distance=7)")
        print("     - Txn3: Normal (distance=10)")
        print("   → 2 fraud votes, 1 normal → Classify as FRAUD")

        print("\n💡 CHOOSING K:")
        print("   Small K: Low bias, high variance (overfitting)")
        print("   Large K: High bias, low variance (underfitting)")
        print("   Rule of thumb: K = √N for classification")

    def _naive_bayes_theory(self):
        """Naive Bayes theory"""
        print("\n" + "=" * 60)
        print("📊 NAIVE BAYES")
        print("=" * 60)

        print("\n🎯 WHAT IT DOES:")
        print("   Uses Bayes' Theorem with 'naive' independence assumption")

        print("\n📐 BAYES' THEOREM:")
        print("   P(Fraud|Features) = P(Features|Fraud) * P(Fraud) / P(Features)")

        print("\n🔍 'NAIVE' ASSUMPTION:")
        print("   Features are independent given the class")
        print("   In reality: amount and location ARE related")
        print("   Despite this, works surprisingly well!")

        print("\n💡 TYPES:")
        print("   Gaussian: For continuous features (amount)")
        print("   Multinomial: For discrete counts (transaction count)")
        print("   Bernoulli: For binary features (night transaction?)")

    def _gradient_boosting_theory(self):
        """Gradient Boosting theory"""
        print("\n" + "=" * 60)
        print("📈 GRADIENT BOOSTING")
        print("=" * 60)

        print("\n🎯 WHAT IT DOES:")
        print("   Sequentially adds trees to correct residuals")

        print("\n🔧 ALGORITHM:")
        print("   1. Start with simple model (mean prediction)")
        print("   2. Calculate residuals (errors)")
        print("   3. Fit tree to predict residuals")
        print("   4. Update model: F = F + learning_rate * tree")
        print("   5. Repeat steps 2-4")

        print("\n📊 XGBOOST vs LIGHTGBM vs CATBOOST:")
        print("   XGBoost: Most popular, handles missing values")
        print("   LightGBM: Faster, less memory, leaf-wise growth")
        print("   CatBoost: Best for categorical features")

    def unsupervised_learning(self):
        """Unsupervised learning for anomaly detection"""
        print("\n" + "=" * 70)
        print("🔍 UNSUPERVISED LEARNING - Anomaly Detection")
        print("=" * 70)

        print("\n🎯 K-MEANS CLUSTERING:")
        print("   Groups similar transactions together")
        print("   Fraud appears as outliers (small clusters)")

        print("\n🎯 DBSCAN (Density-Based):")
        print("   Finds dense regions of normal transactions")
        print("   Fraud = points in low-density regions")
        print("   No need to specify number of clusters")

        print("\n🎯 ISOLATION FOREST:")
        print("   Explicitly designed for anomaly detection")
        print("   Anomalies are 'easier to isolate'")
        print("   Very fast and effective")