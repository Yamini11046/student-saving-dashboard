# app.py
# ==========================================================
# ERROR-FREE STREAMLIT DASHBOARD
# Upload student_savings.csv in same folder
# Run:
# pip install -r requirements.txt
# streamlit run app.py
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ==========================================================
# PAGE SETTINGS
# ==========================================================
st.set_page_config(
    page_title="Student Savings Dashboard",
    page_icon="💰",
    layout="wide"
)

# ==========================================================
# TITLE
# ==========================================================
st.title("💰 Student Saving Capacity Prediction")
st.write("Machine Learning Dashboard")

# ==========================================================
# LOAD DATA SAFELY
# ==========================================================
try:
    df = pd.read_csv("student_savings.csv")
except:
    st.error("student_savings.csv file not found.")
    st.stop()

# ==========================================================
# CLEAN DATA
# ==========================================================
df.fillna(0, inplace=True)

# Encode columns only if exist
for col in df.columns:
    if df[col].dtype == object:
        df[col] = LabelEncoder().fit_transform(df[col])

# ==========================================================
# CHECK REQUIRED COLUMNS
# ==========================================================
required_cols = [
    'Monthly_Income',
    'Food_Expense',
    'Transport_Expense',
    'Entertainment_Expense',
    'Shopping_Expense',
    'Rent_Hostel_Fee',
    'Family_Income',
    'Age',
    'Saving_Class'
]

for col in required_cols:
    if col not in df.columns:
        st.error(f"Column Missing: {col}")
        st.stop()

# ==========================================================
# CREATE TOTAL EXPENSE
# ==========================================================
df["Total_Expense"] = (
    df["Food_Expense"] +
    df["Transport_Expense"] +
    df["Entertainment_Expense"] +
    df["Shopping_Expense"] +
    df["Rent_Hostel_Fee"]
)

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.header("Filters")

age_min = int(df["Age"].min())
age_max = int(df["Age"].max())

age = st.sidebar.slider(
    "Age Range",
    age_min,
    age_max,
    (age_min, age_max)
)

filtered = df[
    (df["Age"] >= age[0]) &
    (df["Age"] <= age[1])
]

# ==========================================================
# MACHINE LEARNING
# ==========================================================
X = filtered[
    [
        'Monthly_Income',
        'Food_Expense',
        'Transport_Expense',
        'Entertainment_Expense',
        'Shopping_Expense',
        'Rent_Hostel_Fee',
        'Family_Income',
        'Age'
    ]
]

y = filtered["Saving_Class"]

# Prevent small dataset crash
if len(filtered) < 5:
    st.warning("Need more rows in dataset.")
    st.stop()

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

# ==========================================================
# KPI
# ==========================================================
c1, c2, c3 = st.columns(3)

c1.metric("Students", len(filtered))
c2.metric("Avg Income", round(filtered["Monthly_Income"].mean(),2))
c3.metric("Accuracy", str(round(acc*100,2))+"%")

# ==========================================================
# GRAPH 1
# ==========================================================
st.subheader("Income vs Expense")

fig1 = px.scatter(
    filtered,
    x="Monthly_Income",
    y="Total_Expense",
    color="Saving_Class"
)

st.plotly_chart(fig1, use_container_width=True)

# ==========================================================
# GRAPH 2
# ==========================================================
st.subheader("Saving Class Distribution")

fig2 = px.histogram(
    filtered,
    x="Saving_Class"
)

st.plotly_chart(fig2, use_container_width=True)

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================
st.subheader("Feature Importance")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

fig3 = px.bar(
    importance,
    x="Feature",
    y="Importance"
)

st.plotly_chart(fig3, use_container_width=True)

# ==========================================================
# CONFUSION MATRIX
# ==========================================================
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, pred)

fig4 = px.imshow(
    cm,
    text_auto=True
)

st.plotly_chart(fig4, use_container_width=True)

# ==========================================================
# FOOTER
# ==========================================================
st.success("Dashboard Running Successfully")
