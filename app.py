import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack

# ---------------- Load model & vectorizer ----------------
svm_model = joblib.load("Models/best_svm.pkl")
tfidf = joblib.load("Artifacts/tfidf_vectorizer.pkl")


# ---------------- Feature extraction ----------------
def extract_features(password):
    length = len(password)
    digits = sum(c.isdigit() for c in password)
    upper = sum(c.isupper() for c in password)
    lower = sum(c.islower() for c in password)
    special = sum(not c.isalnum() for c in password)

    digit_freq = digits / length if length > 0 else 0
    upper_freq = upper / length if length > 0 else 0
    lower_freq = lower / length if length > 0 else 0
    special_freq = special / length if length > 0 else 0

    return np.array([
        length, digits, upper, lower, special, digit_freq, upper_freq, lower_freq, special_freq
    ]).reshape(1, -1)


# ---------------- Softmax for confidence ----------------
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)


# ---------------- UI ----------------
st.set_page_config(
    page_title="Password Strength Classifier",
    page_icon="🔐",
    layout="centered"
)

st.title('Password Strength Classifier')
st.markdown("Enter a password and get a machine learning–based strength evaluation.")

password = st.text_input('Enter your password:')

if password:
    X_text = tfidf.transform([password])
    X_custom = extract_features(password)
    X_final = hstack([X_text, X_custom])

    prediction = svm_model.predict(X_final)[0]

    scores = svm_model.decision_function(X_final)
    probs = softmax(scores)[0]

    label_map = {
        0: "Weak",
        1: "Medium",
        2: "Strong"
    }

    color_map = {
        0: "red",
        1: "yellow",
        2: "green"
    }

    predicted_label = label_map[prediction]
    confidence = probs[prediction]


    # ---------------- Result Card ----------------
    st.markdown("---")
    st.subheader("Result")

    st.markdown(
        f"""
        <div style="padding:20px;border-radius:10px;background-color:#111;">
            <h2 style="color:{color_map[prediction]}">
                Strength: {predicted_label}
            </h2>
            <p>Model confidence: <b>{confidence*100:.2f}%</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )


    # ---------------- Confidence bar ----------------
    st.progress(float(confidence))


    # ---------------- Class probabilities ----------------
    st.subheader("Class Probabilities")

    col1, col2, col3 = st.columns(3)
    col1.metric("Weak", f"{probs[0]*100:.2f}%")
    col2.metric("Medium", f"{probs[1]*100:.2f}%")
    col3.metric("Strong", f"{probs[2]*100:.2f}%")

