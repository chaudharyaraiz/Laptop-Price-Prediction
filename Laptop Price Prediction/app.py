import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

# -------------------------
# 1. Load & Clean Dataset
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("laptop_data.csv")
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df

df = load_data()

st.title("💻 Laptop Price Prediction App")

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# -------------------------
# 2. Preprocess Data
# -------------------------
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("Price", axis=1)
y = df_encoded["Price"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------
# 3. Train or Load Model
# -------------------------
model_file = "laptop_price_model.pkl"

if os.path.exists(model_file):
    model = joblib.load(model_file)
else:
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, model_file)

# -------------------------
# 4. User Input for Prediction
# -------------------------
st.subheader("🖊️ Enter Laptop Features to Predict Price")

# Example: take only numeric columns for simplicity
input_data = {}
for col in X.columns:
    val = st.number_input(f"Enter value for {col}:", value=float(X[col].mean()))
    input_data[col] = val

# Convert user input into DataFrame
input_df = pd.DataFrame([input_data])

# -------------------------
# 5. Make Prediction
# -------------------------
if st.button("🔮 Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Laptop Price: {round(prediction, 2)}")
