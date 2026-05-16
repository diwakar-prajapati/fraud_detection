"""
THEORY MODULE 1: Statistical Learning
Covers: Descriptive stats, Probability, Hypothesis testing, Bayesian inference
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


class StatisticalLearning:
    """Complete statistical learning theory with examples"""

    def __init__(self, df):
        self.df = df
        # Handle both column naming conventions
        if 'Class' in df.columns:
            self.fraud = df[df['Class'] == 1]
            self.normal = df[df['Class'] == 0]
            self.target_col = 'Class'
            self.amount_col = 'Amount' if 'Amount' in df.columns else 'amount'
        else:
            self.fraud = df[df['class'] == 1]
            self.normal = df[df['class'] == 0]
            self.target_col = 'class'
            self.amount_col = 'amount'

    def all_statistical_concepts(self):
        """Demonstrate ALL statistical concepts"""

        print("\n" + "=" * 70)
        print("📊 STATISTICAL LEARNING - COMPLETE THEORY")
        print("=" * 70)

        self._descriptive_stats()
        self._probability_distributions()
        self._hypothesis_testing()
        self._bayesian_inference()
        self._correlation_analysis()
        # Removed _anova_analysis() as it's not critical for fraud detection

    def _descriptive_stats(self):
        """Central tendency, dispersion, shape"""
        print("\n1️⃣ DESCRIPTIVE STATISTICS")
        print("-" * 50)

        # Get amount data
        fraud_amounts = self.fraud[self.amount_col]
        normal_amounts = self.normal[self.amount_col]

        # Central tendency
        print("\n📍 Central Tendency (Where is the center?)")
        metrics = {
            'Mean': fraud_amounts.mean(),
            'Median': fraud_amounts.median(),
            'Mode': fraud_amounts.mode().iloc[0] if len(fraud_amounts.mode()) > 0 else 0
        }
        for name, value in metrics.items():
            print(f"   {name:8s}: ${value:.2f}")

        # Dispersion
        print("\n📈 Dispersion (How spread out?)")
        dispersion = {
            'Variance': fraud_amounts.var(),
            'Std Dev': fraud_amounts.std(),
            'Range': fraud_amounts.max() - fraud_amounts.min(),
            'IQR': fraud_amounts.quantile(0.75) - fraud_amounts.quantile(0.25)
        }
        for name, value in dispersion.items():
            print(f"   {name:8s}: {value:.2f}")

        # Shape
        print("\n📐 Shape of Distribution")
        skewness = fraud_amounts.skew()
        kurtosis = fraud_amounts.kurtosis()
        print(f"   Skewness: {skewness:.3f}")
        print(f"     → {'Right-skewed' if skewness > 0 else 'Left-skewed' if skewness < 0 else 'Symmetric'}")
        print(f"   Kurtosis: {kurtosis:.3f}")
        print(f"     → {'Heavy tails' if kurtosis > 3 else 'Light tails' if kurtosis < 3 else 'Normal-like'}")

    def _probability_distributions(self):
        """All probability distributions"""
        print("\n2️⃣ PROBABILITY DISTRIBUTIONS")
        print("-" * 50)

        fraud_rate = len(self.fraud) / len(self.df)
        fraud_amounts = self.fraud[self.amount_col]
        mean = fraud_amounts.mean()
        std = fraud_amounts.std()

        # Normal Distribution
        print("\n🔔 Normal Distribution (Gaussian)")
        print(f"   Formula: f(x) = (1/(σ√(2π))) * e^(-(x-μ)²/(2σ²))")
        print(f"   Parameters: μ={mean:.2f}, σ={std:.2f}")
        print(f"   68-95-99.7 Rule:")
        print(f"     - 68% within μ±1σ: ${mean - std:.2f} to ${mean + std:.2f}")
        print(f"     - 95% within μ±2σ: ${mean - 2 * std:.2f} to ${mean + 2 * std:.2f}")

        # Poisson Distribution
        print("\n📊 Poisson Distribution (Event Counts)")
        print(f"   λ (lambda) = {fraud_rate:.4f} (frauds per transaction)")
        print(f"   Formula: P(X=k) = (λ^k * e^(-λ)) / k!")
        print(f"   P(0 frauds) = {stats.poisson.pmf(0, fraud_rate):.4f}")
        print(f"   P(1 fraud) = {stats.poisson.pmf(1, fraud_rate):.4f}")

        # Exponential Distribution
        print("\n⏱️ Exponential Distribution (Time Between Events)")
        print(f"   λ (rate) = {fraud_rate:.4f}")
        print(f"   Formula: f(x) = λe^(-λx)")
        print(f"   Memoryless property: P(X > s+t | X > s) = P(X > t)")

        # Binomial Distribution
        print("\n🎲 Binomial Distribution")
        n_trials = 100
        p_success = fraud_rate
        print(f"   n={n_trials} transactions, p={p_success:.4f}")
        print(f"   Formula: P(X=k) = C(n,k) * p^k * (1-p)^(n-k)")
        expected_frauds = n_trials * p_success
        print(f"   Expected frauds in {n_trials} txns: {expected_frauds:.2f}")

    def _hypothesis_testing(self):
        """Statistical hypothesis testing"""
        print("\n3️⃣ HYPOTHESIS TESTING")
        print("-" * 50)

        fraud_amounts = self.fraud[self.amount_col]
        normal_amounts = self.normal[self.amount_col]

        # T-test
        print("\n🔬 T-Test: Fraud Amount vs Normal Amount")
        print("   H₀: Mean fraud amount = Mean normal amount")
        print("   H₁: Mean fraud amount ≠ Mean normal amount")

        t_stat, p_value = stats.ttest_ind(fraud_amounts, normal_amounts)
        print(f"   T-statistic: {t_stat:.4f}")
        print(f"   P-value: {p_value:.6f}")

        alpha = 0.05
        if p_value < alpha:
            print(f"   ✅ REJECT H₀ (p < {alpha})")
            print("   → Fraud amounts ARE significantly different")
        else:
            print(f"   ❌ FAIL TO REJECT H₀ (p >= {alpha})")

        # Confidence Interval
        print("\n📊 95% Confidence Interval")
        ci = stats.t.interval(0.95, len(fraud_amounts) - 1,
                              loc=fraud_amounts.mean(),
                              scale=fraud_amounts.std() / np.sqrt(len(fraud_amounts)))
        print(f"   CI: [${ci[0]:.2f}, ${ci[1]:.2f}]")
        print("   Interpretation: We are 95% confident the true mean falls here")

        # Chi-square test for categorical relationships (if applicable)
        print("\n📈 Chi-Square Test for Independence")
        if 'V1' in self.df.columns:
            # For credit card data, test if V1 differs between fraud/normal
            v1_fraud = self.fraud['V1']
            v1_normal = self.normal['V1']
            # Create bins for chi-square
            bins = np.percentile(self.df['V1'], [0, 25, 50, 75, 100])
            fraud_binned = pd.cut(v1_fraud, bins).value_counts()
            normal_binned = pd.cut(v1_normal, bins).value_counts()
            contingency = pd.DataFrame([fraud_binned, normal_binned])
            chi2, p, dof, expected = stats.chi2_contingency(contingency.fillna(0))
            print(f"   Testing if V1 feature differs between fraud and normal")
            print(f"   Chi-square: {chi2:.4f}, p-value: {p:.6f}")
            if p < 0.05:
                print("   → Feature values ARE different between classes")

    def _bayesian_inference(self):
        """Bayesian statistics"""
        print("\n4️⃣ BAYESIAN INFERENCE")
        print("-" * 50)

        fraud_rate = len(self.fraud) / len(self.df)

        print("\n📐 Bayes' Theorem:")
        print("   P(A|B) = P(B|A) * P(A) / P(B)")

        print("\n🔮 Example: Fraud Detection")
        print(f"   Prior P(Fraud) = {fraud_rate:.4f} (base rate)")
        print(f"   Likelihood P(High Amount|Fraud) = 0.70")
        print(f"   Evidence P(High Amount) = 0.05")

        posterior = (0.70 * fraud_rate) / 0.05
        print(f"   Posterior P(Fraud|High Amount) = {posterior:.4f}")
        print("   → Probability of fraud given high amount")

        # Additional Bayesian example
        print("\n🔮 Example 2: Bayesian Updating")
        prior = fraud_rate
        likelihood = 0.85  # Model accuracy
        evidence = 0.5

        posterior2 = (likelihood * prior) / evidence
        print(f"   Prior: {prior:.4f}")
        print(f"   After evidence: {posterior2:.4f}")

    def _correlation_analysis(self):
        """Correlation and causation"""
        print("\n5️⃣ CORRELATION ANALYSIS")
        print("-" * 50)

        # Get numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        # Limit to first 10 for readability
        numeric_cols = [col for col in numeric_cols if col != self.target_col][:10]

        # Pearson correlation
        correlations = []
        for col in numeric_cols:
            corr = self.df[col].corr(self.df[self.target_col])
            correlations.append((col, corr))

        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        print("\n📊 Pearson Correlation (Linear relationships)")
        print("   Top features correlated with fraud:")
        for feat, corr in correlations[:5]:
            print(f"     {feat}: {corr:.4f}")

        print("\n   Bottom features:")
        for feat, corr in correlations[-3:]:
            print(f"     {feat}: {corr:.4f}")

        print("\n⚠️ Important: Correlation ≠ Causation")
        print("   Example: Ice cream sales and drowning are correlated")
        print("   → Both caused by hot weather, not direct causation")

        # Spearman correlation (monotonic)
        print("\n📊 Spearman Correlation (Monotonic relationships)")
        spearman_corr = self.df[numeric_cols].corrwith(self.df[self.target_col], method='spearman')
        top_spearman = spearman_corr.abs().sort_values(ascending=False).head(3)
        for feat, corr in top_spearman.items():
            print(f"     {feat}: {corr:.4f}")