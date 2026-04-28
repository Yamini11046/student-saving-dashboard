
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Student Savings Dashboard", page_icon="💰", layout="wide")

st.title("💰 Predictive Analysis for Student Saving Capacity")
st.caption("DMW Capstone Project Dashboard")

@st.cache_data
def load_data():
    return pd.read_csv("student_savings.csv")

df = load_data()
df.fillna(df.median(numeric_only=True), inplace=True)

for col in ["Gender","Scholarship","Saving_Class"]:
    if df[col].dtype == object:
        df[col] = LabelEncoder().fit_transform(df[col])

df["Total_Expense"] = (
    df["Food_Expense"] + df["Transport_Expense"] + df["Entertainment_Expense"] +
    df["Shopping_Expense"] + df["Rent_Hostel_Fee"] + df["Mobile_Recharge"]
)

st.sidebar.header("Filters")
gender = st.sidebar.selectbox("Gender", ["All"] + list(df["Gender"].unique()))
age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
age_range = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max))

filtered = df.copy()
if gender != "All":
    filtered = filtered[filtered["Gender"] == gender]
filtered = filtered[(filtered["Age"] >= age_range[0]) & (filtered["Age"] <= age_range[1])]

features = ["Monthly_Income","Food_Expense","Transport_Expense","Entertainment_Expense",
            "Shopping_Expense","Rent_Hostel_Fee","Family_Income","Age"]
X = filtered[features]
y = filtered["Saving_Class"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

c1,c2,c3,c4 = st.columns(4)
c1.metric("Total Students", len(filtered))
c2.metric("Avg Income", round(filtered["Monthly_Income"].mean(),2))
c3.metric("Avg Expense", round(filtered["Total_Expense"].mean(),2))
c4.metric("Accuracy", f"{acc*100:.2f}%")

fig = px.scatter(filtered, x="Monthly_Income", y="Total_Expense", color="Saving_Class", template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)
