"""
Streamlit Dashboard for Fraud Detection - SINGLE PAGE FIXED
Run: streamlit run application/dashboard_single.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import random
from datetime import datetime

# Page config - COMPACT
st.set_page_config(
    page_title="Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Compact CSS
st.markdown("""
<style>
    .block-container { padding-top: 0.5rem; padding-bottom: 0rem; }
    .stMetric { background-color: #f0f2f6; border-radius: 8px; padding: 5px; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.3rem; }
</style>
""", unsafe_allow_html=True)

API_URL = "http://localhost:8000"

# Session state
if 'transactions' not in st.session_state:
    st.session_state.transactions = []
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False


def generate_random_amount():
    """Generate random transaction amount"""
    # Simple random amount instead of lognormal to avoid errors
    return random.uniform(10, 5000)


# ========== HEADER ROW ==========
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

with col1:
    st.markdown("## 🛡️ Fraud Detection")

with col2:
    try:
        r = requests.get(f"{API_URL}/health", timeout=1)
        st.success("✅ API" if r.status_code == 200 else "❌ API")
        api_ok = True
    except:
        st.error("❌ API")
        api_ok = False

with col3:
    if st.button("🔴 LIVE" if not st.session_state.auto_refresh else "⏹️ STOP", use_container_width=True):
        st.session_state.auto_refresh = not st.session_state.auto_refresh

with col4:
    st.caption("XGBoost 99.96%")

st.divider()

# ========== MAIN CONTENT - 3 COLUMNS ==========
col_left, col_mid, col_right = st.columns([1, 1.5, 1])

# ========== LEFT COLUMN: Manual Test ==========
with col_left:
    st.markdown("### ✏️ Test")

    amount = st.number_input("Amount $", min_value=0.01, value=150.0, step=50.0, key="amt")
    hour = st.slider("Hour", 0, 23, 14, key="hr")
    device = st.selectbox("Device", ["mobile", "desktop", "tablet"], key="dev")
    prev = st.number_input("Prev Frauds", 0, 10, 0, key="prev")
    txn_cnt = st.number_input("Txns 24h", 0, 50, 1, key="cnt")

    if st.button("🔍 TEST", type="primary", use_container_width=True) and api_ok:
        try:
            data = {
                "amount": float(amount),
                "time_hour": int(hour),
                "device_type": device,
                "previous_frauds": int(prev),
                "txn_count_last_24h": int(txn_cnt),
                "location_id": 1,
                "merchant_id": 100,
                "days_since_last_txn": 30.0,
                "avg_amount_last_7d": 50.0
            }
            r = requests.post(f"{API_URL}/predict/simple", json=data, timeout=5)
            if r.status_code == 200:
                res = r.json()
                prob = res.get('fraud_probability', 0)
                if prob >= 70:
                    st.error(f"🚨 FRAUD - {prob}%")
                elif prob >= 40:
                    st.warning(f"⚠️ RISK - {prob}%")
                else:
                    st.success(f"✅ SAFE - {prob}%")

                st.session_state.transactions.insert(0, {
                    'timestamp': datetime.now(),
                    'amount': float(amount),
                    'fraud_probability': float(prob),
                    'is_fraud': prob > 50,
                    'transaction_id': res.get('transaction_id', 'TST')[:6]
                })
        except Exception as e:
            st.error(f"API Error: {e}")

# ========== MIDDLE COLUMN: Live Monitor ==========
with col_mid:
    st.markdown("### 📊 Live")

    # Generate auto transaction
    if st.session_state.auto_refresh:
        # FIXED: Use simple random instead of numpy lognormal
        amount_val = random.uniform(10, 5000)
        hour_val = random.randint(0, 23)
        prev_val = random.choices([0, 1, 2], weights=[80, 15, 5])[0]
        txn_cnt_val = random.randint(0, 30)

        score = 0
        if amount_val > 1000:
            score += 40
        elif amount_val > 500:
            score += 20
        if hour_val < 6 or hour_val > 22:
            score += 25
        if prev_val > 0:
            score += 30 * min(prev_val, 3)
        if txn_cnt_val > 10:
            score += 20
        elif txn_cnt_val > 5:
            score += 10
        score += random.randint(-5, 5)
        prob_val = max(0.0, min(float(score), 100.0))

        st.session_state.transactions.insert(0, {
            'timestamp': datetime.now(),
            'amount': round(amount_val, 2),
            'fraud_probability': prob_val,
            'is_fraud': prob_val > 50,
            'transaction_id': f"TXN{random.randint(1000, 9999)}"
        })
        if len(st.session_state.transactions) > 30:
            st.session_state.transactions = st.session_state.transactions[:30]

    df = pd.DataFrame(st.session_state.transactions)

    if len(df) > 0:
        # Mini metrics
        a, b, c, d = st.columns(4)
        with a:
            st.metric("Total", len(df))
        with b:
            fraud_count = int(df['is_fraud'].sum())
            st.metric("Fraud", fraud_count)
        with c:
            avg_amt = float(df['amount'].mean())
            st.metric("Avg $", f"${avg_amt:.0f}")
        with d:
            max_risk = float(df['fraud_probability'].max())
            st.metric("Peak", f"{max_risk:.0f}%")

        # Mini chart
        fig = go.Figure()
        recent_df = df.head(20)
        fig.add_trace(go.Scatter(
            x=recent_df['timestamp'],
            y=recent_df['fraud_probability'],
            mode='lines+markers',
            line=dict(color='red', width=2),
            marker=dict(size=6)
        ))
        fig.add_hline(y=50, line_dash="dash", line_color="orange")
        fig.add_hline(y=70, line_dash="dash", line_color="red")
        fig.update_layout(
            height=200,
            margin=dict(l=0, r=0, t=20, b=0),
            showlegend=False,
            xaxis_title="",
            yaxis_title="Risk %"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Mini table - last 5
        st.markdown("**Recent**")
        for _, row in df.head(5).iterrows():
            risk = "🔴" if row['fraud_probability'] > 70 else "🟡" if row['fraud_probability'] > 40 else "🟢"
            st.text(f"{risk} {row['transaction_id']}  ${row['amount']:.0f}  {row['fraud_probability']:.0f}%")
    else:
        st.info("Click LIVE or TEST to start")

# ========== RIGHT COLUMN: Rules & Status ==========
with col_right:
    st.markdown("### 📋 Rules")
    st.caption("💰 Amount > $1000: +40% risk")
    st.caption("🌙 Late night (12am-6am): +25% risk")
    st.caption("🔄 Previous frauds: +30% each")
    st.caption("📊 >10 transactions/24h: +20% risk")

    st.divider()

    if st.session_state.auto_refresh:
        st.success("🟢 LIVE ACTIVE")
        st.caption("Transactions generating automatically")
    else:
        st.info("⚪ LIVE OFF")
        st.caption("Click LIVE to start auto-monitoring")

    st.divider()
    st.caption(f"📡 API: {API_URL}")
    st.caption("🤖 Model: XGBoost")
    st.caption("📊 Accuracy: 99.96%")

# ========== FOOTER ==========
st.divider()
st.caption("Fraud Detection System | Real-time ML/AI monitoring | Rule-based + XGBoost")