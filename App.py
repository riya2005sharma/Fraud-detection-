import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide",
    page_icon="💳"
)

# ------------------- CUSTOM CSS -------------------
st.markdown("""
<style>
<h1 style='
    text-align: center;
    font-size: 60px;
    font-weight: 900;
    color: #00FFE5;
    text-shadow: 0px 0px 20px rgba(0,255,229,0.8);
'>
💳 Fraud Detection Dashboard
</h1>
h1 {
    animation: fadeIn 2s ease-in;
}
@keyframes fadeIn {
    0% {opacity: 0;}
    100% {opacity: 1;}
}
body {
    background-color: #0e1117;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background: linear-gradient(145deg, #1f2630, #12161d);
    box-shadow: 5px 5px 15px #0b0e12, -5px -5px 15px #232a36;
    text-align: center;
}
.title {
    font-size: 40px;
    font-weight: bold;
    color: #00ffe1;
}
</style>
""", unsafe_allow_html=True)

# ------------------- TITLE -------------------
st.markdown('<p class="title">💳 AI Fraud Detection Dashboard</p>', unsafe_allow_html=True)

# ------------------- LOAD DATA -------------------
df = pd.read_csv("final_fraud_dataset_all_models (1).csv")

# ------------------- SIDEBAR -------------------
st.sidebar.header("🔍 Filters")

risk = st.sidebar.multiselect(
    "Risk Level",
    df['Risk_Level'].unique(),
    default=df['Risk_Level'].unique()
)

model_filter = st.sidebar.multiselect(
    "Select Model",
    ['LR_Pred','DT_Pred','RF_Pred','XGB_Pred','LGB_Pred','ISO_Pred','SVM_Pred','AE_Pred'],
    default=['RF_Pred']
)

df_filtered = df[df['Risk_Level'].isin(risk)]

# ------------------- KPIs -------------------
col1, col2, col3, col4 = st.columns(4)

col1.markdown(f'<div class="card">📊 Total<br><h2>{len(df_filtered)}</h2></div>', unsafe_allow_html=True)
col2.markdown(f'<div class="card">🚨 Fraud<br><h2>{df_filtered["Final_Pred"].sum()}</h2></div>', unsafe_allow_html=True)
col3.markdown(f'<div class="card">⚠️ High Risk<br><h2>{len(df_filtered[df_filtered["Risk_Level"]=="High Risk"])}</h2></div>', unsafe_allow_html=True)
col4.markdown(f'<div class="card">📈 Avg Risk<br><h2>{round(df_filtered["Risk_Score"].mean(),2)}</h2></div>', unsafe_allow_html=True)

st.markdown("---")

# ------------------- FRAUD TREND -------------------
st.subheader("📊 Fraud vs Normal (Animated)")

counts = df_filtered['Final_Pred'].value_counts().reset_index()
counts.columns = ['Type', 'Count']
counts['Type'] = counts['Type'].map({0: 'Normal', 1: 'Fraud'})

fig1 = px.bar(
    counts,
    x='Type',
    y='Count',
    color='Type',
    template="plotly_dark",
    animation_frame='Type'
)
st.plotly_chart(fig1, use_container_width=True)

# ------------------- PIE CHART -------------------
st.subheader("🧠 Risk Distribution")

fig2 = px.pie(
    df_filtered,
    names='Risk_Level',
    hole=0.5,
    color_discrete_sequence=px.colors.sequential.RdBu,
    template="plotly_dark"
)

st.plotly_chart(fig2, use_container_width=True)

# ------------------- MODEL COMPARISON -------------------
st.subheader("🤖 Model Performance Comparison")

model_cols = [
    'LR_Pred','DT_Pred','RF_Pred',
    'XGB_Pred','LGB_Pred','ISO_Pred','SVM_Pred','AE_Pred'
]

model_sum = df_filtered[model_cols].sum().reset_index()
model_sum.columns = ['Model', 'Fraud_Count']

fig3 = px.bar(
    model_sum,
    x='Model',
    y='Fraud_Count',
    color='Fraud_Count',
    template="plotly_dark"
)

st.plotly_chart(fig3, use_container_width=True)

# ------------------- HEATMAP -------------------
st.subheader("🔥 Correlation Heatmap")

corr = df_filtered[model_cols].corr()

fig4 = go.Figure(data=go.Heatmap(
    z=corr.values,
    x=corr.columns,
    y=corr.columns,
    colorscale='RdBu'
))

fig4.update_layout(template="plotly_dark")
st.plotly_chart(fig4, use_container_width=True)

# ------------------- SCATTER -------------------
st.subheader("📍 Transaction Risk Analysis")

fig5 = px.scatter(
    df_filtered,
    x='Amount',
    y='Risk_Score',
    color='Risk_Level',
    size='Risk_Score',
    hover_data=['Final_Pred'],
    template="plotly_dark"
)

st.plotly_chart(fig5, use_container_width=True)

# ------------------- LIVE SIMULATION -------------------
st.subheader("⚡ Live Fraud Simulation")

if st.button("Start Simulation"):
    import time
    for i in range(5):
        sample = df.sample(1)
        risk_val = sample['Risk_Score'].values[0]

        if risk_val > 0.7:
            st.error("🚨 Fraud Detected!")
        else:
            st.success("✅ Safe Transaction")

        time.sleep(1)

# ------------------- DATA TABLE -------------------
st.subheader("📋 Data Preview")
st.dataframe(df_filtered.sample(100))