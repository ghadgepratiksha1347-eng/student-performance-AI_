# ============================================================
# STUDENT PERFORMANCE AI PREDICTOR
# NumPy + Pandas + Matplotlib + Scikit-learn + Streamlit
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# 2. TITLE
# ============================================================

st.title("🎓 Student Performance AI Predictor")

st.write(
    "A Machine Learning project using NumPy, Pandas, "
    "Matplotlib, Scikit-learn and Streamlit."
)

st.divider()


# ============================================================
# 3. CREATE DATASET USING PANDAS
# ============================================================

data = {
    "Name": [
        "Amit", "Priya", "Rahul", "Sneha", "Pooja",
        "Akash", "Neha", "Rohit", "Kiran", "Om",
        "Riya", "Vikas", "Anjali", "Sahil", "Meena"
    ],

    "Study_Hours": [
        3, 5, 2, 6, 4,
        3, 7, 2, 5, 4,
        6, 2, 5, 3, 7
    ],

    "Attendance": [
        70, 90, 65, 95, 85,
        75, 98, 60, 88, 80,
        92, 68, 89, 72, 96
    ],

    "Previous_Marks": [
        55, 78, 50, 88, 72,
        60, 92, 45, 82, 68,
        86, 52, 80, 58, 94
    ],

    "Passed": [
        "No", "Yes", "No", "Yes", "Yes",
        "No", "Yes", "No", "Yes", "Yes",
        "Yes", "No", "Yes", "No", "Yes"
    ]
}

df = pd.DataFrame(data)


# ============================================================
# 4. SHOW DATASET
# ============================================================

st.subheader("📋 Student Dataset")

st.dataframe(
    df,
    use_container_width=True
)


# ============================================================
# 5. FEATURES AND TARGET
# ============================================================

X = df[
    [
        "Study_Hours",
        "Attendance",
        "Previous_Marks"
    ]
]

y = df["Passed"]


# ============================================================
# 6. CONVERT PASS/FAIL TO 0/1
# ============================================================

y_encoded = y.map({
    "No": 0,
    "Yes": 1
})


# ============================================================
# 7. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


# ============================================================
# 8. STANDARD SCALER
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ============================================================
# 9. LOGISTIC REGRESSION MODEL
# ============================================================

model = LogisticRegression()

model.fit(
    X_train_scaled,
    y_train
)


# ============================================================
# 10. MODEL PREDICTION
# ============================================================

y_prediction = model.predict(
    X_test_scaled
)


# ============================================================
# 11. MODEL ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_prediction
)


# ============================================================
# 12. SIDEBAR INPUT
# ============================================================

st.sidebar.header("🎯 Enter Student Details")

study_hours = st.sidebar.slider(
    "Study Hours",
    0,
    12,
    5
)

attendance = st.sidebar.slider(
    "Attendance (%)",
    0,
    100,
    80
)

previous_marks = st.sidebar.slider(
    "Previous Marks",
    0,
    100,
    70
)


# ============================================================
# 13. NEW STUDENT DATA
# ============================================================

new_student = pd.DataFrame({
    "Study_Hours": [study_hours],
    "Attendance": [attendance],
    "Previous_Marks": [previous_marks]
})


# ============================================================
# 14. SCALE NEW DATA
# ============================================================

new_student_scaled = scaler.transform(
    new_student
)


# ============================================================
# 15. PREDICT PASS / FAIL
# ============================================================

prediction = model.predict(
    new_student_scaled
)


# ============================================================
# 16. PREDICTION PROBABILITY
# ============================================================

probability = model.predict_proba(
    new_student_scaled
)


# ============================================================
# 17. DISPLAY STUDENT INPUT
# ============================================================

st.subheader("👨‍🎓 Student Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Study Hours",
        study_hours
    )

with col2:
    st.metric(
        "Attendance",
        f"{attendance}%"
    )

with col3:
    st.metric(
        "Previous Marks",
        previous_marks
    )


# ============================================================
# 18. FINAL PREDICTION
# ============================================================

st.divider()

st.subheader("🤖 AI Prediction")


if prediction[0] == 1:

    pass_probability = probability[0][1] * 100

    st.success(
        "🎉 RESULT: STUDENT WILL PASS"
    )

    st.metric(
        "Pass Probability",
        f"{pass_probability:.2f}%"
    )

else:

    fail_probability = probability[0][0] * 100

    st.error(
        "⚠️ RESULT: STUDENT MAY FAIL"
    )

    st.metric(
        "Fail Probability",
        f"{fail_probability:.2f}%"
    )


# ============================================================
# 19. MODEL ACCURACY
# ============================================================

st.divider()

st.subheader("📊 Machine Learning Model Performance")

st.metric(
    "Model Accuracy",
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# 20. CONFUSION MATRIX
# ============================================================

st.subheader("🔲 Confusion Matrix")

cm = confusion_matrix(
    y_test,
    y_prediction
)

fig_cm, ax_cm = plt.subplots()

ax_cm.imshow(cm)

ax_cm.set_title("Confusion Matrix")

ax_cm.set_xlabel("Predicted")

ax_cm.set_ylabel("Actual")

for i in range(len(cm)):
    for j in range(len(cm[i])):
        ax_cm.text(
            j,
            i,
            cm[i][j],
            ha="center",
            va="center"
        )

st.pyplot(fig_cm)


# ============================================================
# 21. CLASSIFICATION REPORT
# ============================================================

st.subheader("📄 Classification Report")

report = classification_report(
    y_test,
    y_prediction,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df,
    use_container_width=True
)


# ============================================================
# 22. NUMPY STATISTICS
# ============================================================

st.divider()

st.subheader("🔢 Student Statistics")

average_marks = np.mean(
    df["Previous_Marks"]
)

maximum_marks = np.max(
    df["Previous_Marks"]
)

minimum_marks = np.min(
    df["Previous_Marks"]
)

median_marks = np.median(
    df["Previous_Marks"]
)

std_marks = np.std(
    df["Previous_Marks"]
)


col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Average",
        f"{average_marks:.2f}"
    )

with col2:
    st.metric(
        "Maximum",
        maximum_marks
    )

with col3:
    st.metric(
        "Minimum",
        minimum_marks
    )

with col4:
    st.metric(
        "Median",
        f"{median_marks:.2f}"
    )

with col5:
    st.metric(
        "Std Dev",
        f"{std_marks:.2f}"
    )


# ============================================================
# 23. GRAPH - STUDY HOURS VS MARKS
# ============================================================

st.divider()

st.subheader("📈 Study Hours vs Previous Marks")

fig1, ax1 = plt.subplots()

ax1.scatter(
    df["Study_Hours"],
    df["Previous_Marks"]
)

ax1.set_xlabel("Study Hours")

ax1.set_ylabel("Previous Marks")

ax1.set_title(
    "Study Hours vs Previous Marks"
)

ax1.grid(True)

st.pyplot(fig1)


# ============================================================
# 24. GRAPH - STUDENT MARKS
# ============================================================

st.subheader("📊 Student Previous Marks")

fig2, ax2 = plt.subplots()

ax2.bar(
    df["Name"],
    df["Previous_Marks"]
)

ax2.set_xlabel("Student")

ax2.set_ylabel("Marks")

ax2.set_title(
    "Student Previous Marks"
)

plt.xticks(
    rotation=45
)

st.pyplot(fig2)


# ============================================================
# 25. GRAPH - ATTENDANCE
# ============================================================

st.subheader("📊 Student Attendance")

fig3, ax3 = plt.subplots()

ax3.bar(
    df["Name"],
    df["Attendance"]
)

ax3.set_xlabel("Student")

ax3.set_ylabel("Attendance %")

ax3.set_title(
    "Student Attendance"
)

plt.xticks(
    rotation=45
)

st.pyplot(fig3)
