import numpy as np
import streamlit as st

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(page_title="Job Recommendation System", layout="centered")
st.title("💼 Job Recommendation System using ANN")

# -------------------------------
# Activation Function
# -------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# -------------------------------
# Training Data
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

# Normalize (0–1)
X[:, :5] /= 10

# Job labels
y = np.array([
    [1,0,0,0],  # Software Dev
    [0,1,0,0],  # Data Scientist
    [0,0,1,0],  # Web Dev
    [0,0,0,1],  # AI/ML Engineer
    [0,0,1,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1]
])

# -------------------------------
# Initialize Weights
# -------------------------------
np.random.seed(1)
W1 = np.random.randn(6, 8)
W2 = np.random.randn(8, 4)

# -------------------------------
# Train ANN (Backpropagation)
# -------------------------------
lr = 0.1
for _ in range(3000):
    h = sigmoid(np.dot(X, W1))
    o = sigmoid(np.dot(h, W2))

    error = y - o
    d2 = error * o * (1 - o)
    d1 = d2.dot(W2.T) * h * (1 - h)

    W2 += h.T.dot(d2) * lr
    W1 += X.T.dot(d1) * lr

# -------------------------------
# User Input
# -------------------------------
st.subheader("🔹 Enter Your Skills")

coding = st.slider("Coding Skill", 0, 10, 5)
math = st.slider("Math Skill", 0, 10, 5)
ml = st.slider("ML Knowledge", 0, 10, 5)
web = st.slider("Web Knowledge", 0, 10, 5)
exp = st.slider("Experience (Years)", 0, 10, 1)
degree = st.selectbox("Degree", ["Non-CS", "CS / IT"])

degree = 1 if degree == "CS / IT" else 0

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Recommend Job"):
    user = np.array([[coding, math, ml, web, exp, degree]], dtype=float)
    user[:, :5] /= 10

    hidden = sigmoid(np.dot(user, W1))
    output = sigmoid(np.dot(hidden, W2))

    job_index = np.argmax(output)

    jobs = [
        "Software Developer",
        "Data Scientist",
        "Web Developer",
        "AI/ML Engineer"
    ]

    st.success(f"✅ Recommended Job Role: **{jobs[job_index]}**")

st.markdown("---")
st.caption("Mini Project | ANN from Scratch | Streamlit Cloud")
