import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# ------------------ Load Files ------------------
model = load_model("model.h5")

with open("gender_encoding.pkl", "rb") as f:
    gender_encoder = pickle.load(f)

with open("geography_encoding.pkl", "rb") as f:
    geo_encoder = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ------------------ UI CONFIG ------------------
st.set_page_config(page_title="Churn Prediction", page_icon="💳", layout="wide")

st.title("💳 Customer Churn Prediction")
st.markdown("### Predict whether a customer will leave the bank")

# ------------------ INPUT UI ------------------
col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Credit Score", 300, 900, 600)
    geography = st.selectbox("Geography", geo_encoder.categories_[0])
    gender = st.selectbox("Gender", gender_encoder.classes_)
    age = st.slider("Age", 18, 100, 30)
    tenure = st.slider("Tenure (years)", 0, 10, 3)

with col2:
    balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_cr_card = st.selectbox("Has Credit Card", [0, 1])
    is_active_member = st.selectbox("Is Active Member", [0, 1])
    estimated_salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

# ------------------ PREPROCESS ------------------

# Encode gender
gender_encoded = gender_encoder.transform([gender])[0]

# Encode geography (OneHot)
geo_encoded = geo_encoder.transform([[geography]]).toarray()

# 🔥 IMPORTANT: Correct feature order (same as training)
# First numeric + label encoded → then OneHot

input_data = np.array([[credit_score,
                        gender_encoded,
                        age,
                        tenure,
                        balance,
                        num_products,
                        has_cr_card,
                        is_active_member,
                        estimated_salary]])

# Append geo encoding at END (IMPORTANT)
input_data = np.concatenate([input_data, geo_encoded], axis=1)

# Scale
input_scaled = scaler.transform(input_data)

# ------------------ PREDICTION ------------------

if st.button("🚀 Predict Churn"):

    prediction = model.predict(input_scaled)
    prob = float(prediction[0][0])

    st.subheader("📊 Prediction Result")

    if prob > 0.5:
        st.error(f"⚠️ Customer is likely to churn (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Customer is likely to stay (Probability: {prob:.2f})")

    st.progress(prob)

    # ------------------ DEBUG (remove later) ------------------
    with st.expander("🔍 Debug Info"):
        st.write("Input Data:", input_data)
        st.write("Scaled Data:", input_scaled)
        st.write("Raw Prediction:", prediction)
        st.write("Probability:", prob)

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("🔥 Built with Streamlit | Aman Singh ML Project")