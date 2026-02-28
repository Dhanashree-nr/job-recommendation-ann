import numpy as np
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(page_title="Job Recommendation System", layout="centered")
st.title("💼 Job Recommendation System using ANN")

# -------------------------------
# Training Dataset
# Features: [coding, math, ml, web, experience, degree]
# -------------------------------
X = np.array([
    [8, 6, 2, 3, 2, 1],
    [9, 9, 8, 2, 3, 1],
    [6, 3, 1, 9, 1, 1],
    [8, 8, 9, 3, 2, 1],
    [7, 6, 3, 8, 2, 1],
    [9, 8, 7, 2, 4, 1],
    [5, 4, 1, 7, 1, 0],
    [8, 9, 8, 3, 3, 1]
], dtype=float)

# Normalize manually (0–1 range)
X[:, :5] = X[:, :5] / 10.0

y = np.array([0, 1, 2, 3, 2, 1, 2, 3])
y = to_categorical(y)

# -------------------------------
# ANN Model
# -------------------------------
model = Sequential([
    Dense(16, activation='relu', input_shape=(6,)),
    Dense(8, activation='relu'),
    Dense(4, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X, y, epochs=200, verbose=0)

# -------------------------------
# User Input
# -------------------------------
st.subheader("🔹 Enter Your Skills")

coding = st.slider("Coding Skill (0–10)", 0, 10, 5)
math = st.slider("Math Skill (0–10)", 0, 10, 5)
ml = st.slider("Machine Learning Knowledge (0–10)", 0, 10, 5)
web = st.slider("Web Development Knowledge (0–10)", 0, 10, 5)
exp = st.slider("Experience (Years)", 0, 10, 1)
degree = st.selectbox("Degree Background", ["Non-CS", "CS / IT"])

degree = 1 if degree == "CS / IT" else 0

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Recommend Job"):
    user_input = np.array([[coding, math, ml, web, exp, degree]], dtype=float)
    user_input[:, :5] = user_input[:, :5] / 10.0

    prediction = model.predict(user_input)
    job_index = np.argmax(prediction)

    jobs = [
        "Software Developer",
        "Data Scientist",
        "Web Developer",
        "AI/ML Engineer"
    ]

    st.success(f"✅ Recommended Job Role: **{jobs[job_index]}**")

st.markdown("---")
st.caption("Mini Project | Artificial Neural Network | Streamlit Cloud")
