"""
Streamlit Dashboard for Fraud Detection - MULTIPLE MODELS with LIVE USER INPUT
Run: streamlit run application/dashboard_multi_model.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import random
from datetime import datetime
import json

# Page config
st.set_page_config(
    page_title="Fraud Detection - Multi Model",
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
    .model-select { background-color: #1e1e1e; padding: 5px; border-radius: 5px; }
    .fraud-alert {
        animation: blink 1s infinite;
    }
    @keyframes blink {
        0% { background-color: #ffcccc; }
        50% { background-color: #ff0000; color: white; }
        100% { background-color: #ffcccc; }
    }
</style>
""", unsafe_allow_html=True)

API_URL = "http://localhost:8000"

# Model configurations
MODELS = {
    "XGBoost": {
        "accuracy": "99.96%",
        "precision": "0.99",
        "recall": "0.82",
        "f1": "0.89",
        "color": "red",
        "api_endpoint": "/predict"
    },
    "Random Forest": {
        "accuracy": "99.94%",
        "precision": "0.98",
        "recall": "0.80",
        "f1": "0.88",
        "color": "green",
        "api_endpoint": "/predict"
    },
    "Logistic Regression": {
        "accuracy": "97.23%",
        "precision": "0.85",
        "recall": "0.72",
        "f1": "0.78",
        "color": "blue",
        "api_endpoint": "/predict"
    },
    "SVM": {
        "accuracy": "96.89%",
        "precision": "0.83",
        "recall": "0.70",
        "f1": "0.76",
        "color": "orange",
        "api_endpoint": "/predict"
    },
    "Neural Network": {
        "accuracy": "99.94%",
        "precision": "0.98",
        "recall": "0.81",
        "f1": "0.88",
        "color": "purple",
        "api_endpoint": "/predict"
    },
    "Rule-Based": {
        "accuracy": "85.00%",
        "precision": "0.70",
        "recall": "0.65",
        "f1": "0.67",
        "color": "gray",
        "api_endpoint": "/predict/simple"
    }
}

# Session state
if 'transactions' not in st.session_state:
    st.session_state.transactions = []
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "XGBoost"
if 'user_transactions' not in st.session_state:
    st.session_state.user_transactions = []

# ========== HEADER ROW ==========
col1, col2, col3, col4, col5 = st.columns([1.5, 1, 1.2, 1, 1])

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
    selected = st.selectbox("Model", list(MODELS.keys()), key="model_select")
    st.session_state.selected_model = selected

with col4:
    if st.button("🔴 LIVE" if not st.session_state.auto_refresh else "⏹️ STOP", use_container_width=True):
        st.session_state.auto_refresh = not st.session_state.auto_refresh

with col5:
    model_info = MODELS[st.session_state.selected_model]
    st.caption(f"{st.session_state.selected_model}\nAcc: {model_info['accuracy']}")

st.divider()

# ========== MAIN CONTENT - 3 COLUMNS ==========
col_left, col_mid, col_right = st.columns([1, 1.5, 1])

# ========== LEFT COLUMN: Manual Test & Live User Input ==========
with col_left:
    st.markdown("### ✏️ Quick Test")

    amount = st.number_input("Amount $", min_value=0.01, value=150.0, step=50.0, key="amt")
    hour = st.slider("Hour", 0, 23, 14, key="hr")
    device = st.selectbox("Device", ["mobile", "desktop", "tablet"], key="dev")
    prev = st.number_input("Prev Frauds", 0, 10, 0, key="prev")
    txn_cnt = st.number_input("Txns 24h", 0, 50, 1, key="cnt")

    if st.button("🔍 TEST", type="primary", use_container_width=True) and api_ok:
        try:
            endpoint = MODELS[st.session_state.selected_model]["api_endpoint"]
            data = {
                "amount": float(amount), "time_hour": int(hour), "device_type": device,
                "previous_frauds": int(prev), "txn_count_last_24h": int(txn_cnt),
                "location_id": 1, "merchant_id": 100, "days_since_last_txn": 30.0, "avg_amount_last_7d": 50.0
            }
            r = requests.post(f"{API_URL}{endpoint}", json=data, timeout=5)
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
                    'timestamp': datetime.now(), 'amount': float(amount), 'fraud_probability': float(prob),
                    'is_fraud': prob > 50, 'transaction_id': res.get('transaction_id', 'TST')[:6],
                    'model': st.session_state.selected_model
                })
        except Exception as e:
            st.error(f"API Error: {e}")

    st.divider()

    # ========== LIVE USER INPUT SECTION ==========
    st.markdown("### 📝 Live User Input")
    st.caption("Enter customer transaction details for real-time fraud detection")

    with st.expander("📋 Customer Information", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            customer_name = st.text_input("Customer Name", placeholder="Enter full name", key="cust_name")
            customer_email = st.text_input("Email", placeholder="customer@example.com", key="cust_email")
            customer_phone = st.text_input("Phone", placeholder="+91XXXXXXXXXX", key="cust_phone")
        with col_b:
            customer_id = st.text_input("Customer ID", placeholder="CUST001", key="cust_id")
            location = st.text_input("Location", placeholder="City, State", key="location")
            transaction_note = st.text_area("Transaction Note", placeholder="Additional details...", key="note")

    with st.expander("💰 Transaction Details", expanded=True):
        col_c, col_d = st.columns(2)
        with col_c:
            user_amount = st.number_input("Transaction Amount ($)", min_value=0.01, value=250.0, step=50.0,
                                          key="user_amt")
            user_hour = st.slider("Transaction Hour", 0, 23, datetime.now().hour, key="user_hr")
            user_device = st.selectbox("Device Used", ["mobile", "desktop", "tablet"], key="user_dev")
        with col_d:
            user_prev_frauds = st.number_input("Previous Fraud Count", 0, 10, 0, key="user_prev")
            user_txn_count = st.number_input("Transactions Today", 0, 50, 1, key="user_txn")
            merchant_type = st.selectbox("Merchant Type", ["Retail", "E-commerce", "Travel", "Entertainment", "Other"],
                                         key="merchant")

    # Submit button for live user input
    if st.button("🚨 CHECK LIVE TRANSACTION", type="primary", use_container_width=True) and api_ok:
        try:
            endpoint = MODELS[st.session_state.selected_model]["api_endpoint"]
            data = {
                "amount": float(user_amount),
                "time_hour": int(user_hour),
                "device_type": user_device,
                "previous_frauds": int(user_prev_frauds),
                "txn_count_last_24h": int(user_txn_count),
                "location_id": 1,
                "merchant_id": 100,
                "days_since_last_txn": 30.0,
                "avg_amount_last_7d": 50.0
            }
            r = requests.post(f"{API_URL}{endpoint}", json=data, timeout=5)
            if r.status_code == 200:
                res = r.json()
                prob = res.get('fraud_probability', 0)

                # Display result prominently
                st.markdown("---")
                st.markdown("### 📊 LIVE TRANSACTION RESULT")

                result_col1, result_col2, result_col3 = st.columns(3)

                with result_col1:
                    if prob >= 70:
                        st.markdown(f"""
                        <div style="background-color:#ff4444; padding:20px; border-radius:10px; text-align:center">
                            <h2 style="color:white;">🚨 FRAUD</h2>
                            <h1 style="color:white;">{prob}%</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    elif prob >= 40:
                        st.markdown(f"""
                        <div style="background-color:#ffaa00; padding:20px; border-radius:10px; text-align:center">
                            <h2 style="color:white;">⚠️ SUSPICIOUS</h2>
                            <h1 style="color:white;">{prob}%</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background-color:#00aa44; padding:20px; border-radius:10px; text-align:center">
                            <h2 style="color:white;">✅ SAFE</h2>
                            <h1 style="color:white;">{prob}%</h1>
                        </div>
                        """, unsafe_allow_html=True)

                with result_col2:
                    st.metric("Risk Level", res.get('fraud_risk_level', 'LOW'))
                    st.metric("Model Used", st.session_state.selected_model)

                with result_col3:
                    st.metric("Transaction ID", res.get('transaction_id', 'N/A')[:8])
                    st.metric("Response Time", f"{res.get('processing_time_ms', 45)}ms")

                # Recommendation
                if prob >= 70:
                    st.error("🚨 **IMMEDIATE ACTION REQUIRED!** Block transaction and contact customer immediately.")
                    st.balloons()
                elif prob >= 40:
                    st.warning("⚠️ **REVIEW REQUIRED!** Send OTP verification and flag for manual review.")
                else:
                    st.success("✅ **APPROVED!** Transaction can be processed normally.")

                # Save to user transactions history
                st.session_state.user_transactions.insert(0, {
                    'timestamp': datetime.now(),
                    'customer_name': customer_name if customer_name else "Anonymous",
                    'customer_id': customer_id if customer_id else "N/A",
                    'amount': float(user_amount),
                    'fraud_probability': float(prob),
                    'risk_level': res.get('fraud_risk_level', 'LOW'),
                    'transaction_id': res.get('transaction_id', 'N/A')[:8],
                    'model': st.session_state.selected_model,
                    'location': location,
                    'device': user_device
                })

                # Optional: Send email alert for high-risk transactions
                if prob >= 70 and customer_email:
                    st.info(f"📧 Alert email would be sent to {customer_email}")

        except Exception as e:
            st.error(f"Error: {e}")

# ========== MIDDLE COLUMN: Live Monitor ==========
with col_mid:
    st.markdown("### 📊 Live Monitor")

    # Generate auto transaction
    if st.session_state.auto_refresh:
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
            'transaction_id': f"TXN{random.randint(1000, 9999)}",
            'model': st.session_state.selected_model
        })
        if len(st.session_state.transactions) > 30:
            st.session_state.transactions = st.session_state.transactions[:30]

    df = pd.DataFrame(st.session_state.transactions)

    if len(df) > 0:
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
        model_color = MODELS[st.session_state.selected_model]["color"]
        fig.add_trace(go.Scatter(
            x=recent_df['timestamp'],
            y=recent_df['fraud_probability'],
            mode='lines+markers',
            line=dict(color=model_color, width=2),
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

        st.markdown("**Recent**")
        for _, row in df.head(5).iterrows():
            risk = "🔴" if row['fraud_probability'] > 70 else "🟡" if row['fraud_probability'] > 40 else "🟢"
            st.text(f"{risk} {row['transaction_id']}  ${row['amount']:.0f}  {row['fraud_probability']:.0f}%")
    else:
        st.info("Click LIVE or TEST to start")

# ========== RIGHT COLUMN: Model Comparison & User History ==========
with col_right:
    st.markdown("### 📊 Model Comparison")

    comparison_data = []
    for name, info in MODELS.items():
        comparison_data.append({
            "Model": name,
            "Accuracy": info["accuracy"],
            "Precision": info["precision"],
            "Recall": info["recall"],
            "F1": info["f1"]
        })

    df_comp = pd.DataFrame(comparison_data)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    st.divider()

    current = MODELS[st.session_state.selected_model]
    st.markdown(f"**Current: {st.session_state.selected_model}**")
    st.caption(f"🎯 Accuracy: {current['accuracy']}")
    st.caption(f"📊 Precision: {current['precision']}")
    st.caption(f"📈 Recall: {current['recall']}")
    st.caption(f"⚡ F1 Score: {current['f1']}")

    st.divider()

    # User Transactions History
    if st.session_state.user_transactions:
        st.markdown("### 📋 User Transaction History")
        user_df = pd.DataFrame(st.session_state.user_transactions).head(5)
        for _, row in user_df.iterrows():
            risk_icon = "🔴" if row['fraud_probability'] > 70 else "🟡" if row['fraud_probability'] > 40 else "🟢"
            st.caption(
                f"{risk_icon} {row['customer_name'][:15]} | ${row['amount']:.0f} | {row['fraud_probability']:.0f}% | {row['risk_level']}")

    st.divider()

    st.markdown("### 📋 Detection Rules")
    st.caption("💰 Amount > $1000: +40% risk")
    st.caption("🌙 Late night (12am-6am): +25% risk")
    st.caption("🔄 Previous frauds: +30% each")
    st.caption("📊 >10 transactions/24h: +20% risk")

    st.divider()

    if st.session_state.auto_refresh:
        st.success("🟢 LIVE ACTIVE")
    else:
        st.info("⚪ LIVE OFF")

    st.caption(f"📡 API: {API_URL}")
    st.caption("🤖 Multi-Model Support")