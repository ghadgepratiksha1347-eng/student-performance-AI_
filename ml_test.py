# ============================================================
# SCikit-learn - ALL IMPORTANT CONCEPTS COMBINED
# ============================================================

# Install first:
# pip install scikit-learn pandas numpy matplotlib joblib

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 1. DATASET
# ------------------------------------------------------------

data = {
    "Name": [
        "Amit", "Priya", "Rahul", "Sneha", "Pooja",
        "Akash", "Neha", "Rohit", "Kiran", "Om"
    ],

    "Gender": [
        "Male", "Female", "Male", "Female", "Female",
        "Male", "Female", "Male", "Female", "Male"
    ],

    "Study_Hours": [3, 5, 2, 6, 4, 3, 7, 2, 5, 4],

    "Attendance": [70, 90, 65, 95, 85, 75, 98, 60, 88, 80],

    "Previous_Marks": [55, 78, 50, 88, 72, 60, 92, 45, 82, 68],

    "Passed": [
        "No", "Yes", "No", "Yes", "Yes",
        "No", "Yes", "No", "Yes", "Yes"
    ]
}

df = pd.DataFrame(data)

print("\n========== DATASET ==========")
print(df)


# ------------------------------------------------------------
# 2. FEATURES (X) AND TARGET (y)
# ------------------------------------------------------------

X = df[["Study_Hours", "Attendance", "Previous_Marks"]]

y = df["Passed"]

print("\n========== FEATURES (X) ==========")
print(X)

print("\n========== TARGET (y) ==========")
print(y)


# ------------------------------------------------------------
# 3. LABEL ENCODING
# ------------------------------------------------------------

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\n========== LABEL ENCODING ==========")
print(y_encoded)

# No = 0
# Yes = 1


# ------------------------------------------------------------
# 4. TRAIN TEST SPLIT
# ------------------------------------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

print("\n========== TRAIN TEST SPLIT ==========")

print("Training data:")
print(X_train)

print("\nTesting data:")
print(X_test)


# ------------------------------------------------------------
# 5. STANDARD SCALER
# ------------------------------------------------------------

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("\n========== STANDARD SCALER ==========")
print("Scaled Training Data:")
print(X_train_scaled)

print("\nScaled Testing Data:")
print(X_test_scaled)


# ============================================================
# 6. LOGISTIC REGRESSION
# ============================================================

from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression()

logistic_model.fit(X_train_scaled, y_train)

logistic_prediction = logistic_model.predict(X_test_scaled)

print("\n========== LOGISTIC REGRESSION ==========")
print("Prediction:", logistic_prediction)


# ------------------------------------------------------------
# 7. ACCURACY
# ------------------------------------------------------------

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(
    y_test,
    logistic_prediction
)

print("Accuracy:", accuracy)


# ------------------------------------------------------------
# 8. CONFUSION MATRIX
# ------------------------------------------------------------

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    y_test,
    logistic_prediction
)

print("\n========== CONFUSION MATRIX ==========")
print(cm)


# ------------------------------------------------------------
# 9. CLASSIFICATION REPORT
# ------------------------------------------------------------

from sklearn.metrics import classification_report

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        logistic_prediction,
        zero_division=0
    )
)


# ============================================================
# 10. DECISION TREE
# ============================================================

from sklearn.tree import DecisionTreeClassifier

tree_model = DecisionTreeClassifier(
    random_state=42
)

tree_model.fit(X_train, y_train)

tree_prediction = tree_model.predict(X_test)

print("\n========== DECISION TREE ==========")
print("Prediction:", tree_prediction)

print(
    "Accuracy:",
    accuracy_score(y_test, tree_prediction)
)


# ============================================================
# 11. RANDOM FOREST
# ============================================================

from sklearn.ensemble import RandomForestClassifier

forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

forest_model.fit(X_train, y_train)

forest_prediction = forest_model.predict(X_test)

print("\n========== RANDOM FOREST ==========")
print("Prediction:", forest_prediction)

print(
    "Accuracy:",
    accuracy_score(y_test, forest_prediction)
)


# ============================================================
# 12. KNN
# ============================================================

from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(
    n_neighbors=3
)

knn_model.fit(X_train_scaled, y_train)

knn_prediction = knn_model.predict(X_test_scaled)

print("\n========== KNN ==========")
print("Prediction:", knn_prediction)

print(
    "Accuracy:",
    accuracy_score(y_test, knn_prediction)
)


# ============================================================
# 13. LINEAR REGRESSION
# ============================================================

# Numerical prediction example:
# Predict final marks

from sklearn.linear_model import LinearRegression

X_marks = df[[
    "Study_Hours",
    "Attendance"
]]

y_marks = np.array([
    60, 78, 55, 90, 75,
    62, 95, 50, 85, 72
])

X_marks_train, X_marks_test, y_marks_train, y_marks_test = train_test_split(
    X_marks,
    y_marks,
    test_size=0.2,
    random_state=42
)

linear_model = LinearRegression()

linear_model.fit(
    X_marks_train,
    y_marks_train
)

marks_prediction = linear_model.predict(
    X_marks_test
)

print("\n========== LINEAR REGRESSION ==========")

print("Actual Marks:")
print(y_marks_test)

print("\nPredicted Marks:")
print(marks_prediction)


# ------------------------------------------------------------
# 14. REGRESSION METRICS
# ------------------------------------------------------------

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

mae = mean_absolute_error(
    y_marks_test,
    marks_prediction
)

mse = mean_squared_error(
    y_marks_test,
    marks_prediction
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_marks_test,
    marks_prediction
)

print("\n========== REGRESSION METRICS ==========")

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)


# ============================================================
# 15. NEW STUDENT PREDICTION
# ============================================================

new_student = pd.DataFrame({
    "Study_Hours": [6],
    "Attendance": [92],
    "Previous_Marks": [85]
})

new_student_scaled = scaler.transform(new_student)

new_prediction = logistic_model.predict(
    new_student_scaled
)

print("\n========== NEW STUDENT ==========")

print(new_student)

if new_prediction[0] == 1:
    print("Prediction: PASS")
else:
    print("Prediction: FAIL")


# ============================================================
# 16. PREDICTION PROBABILITY
# ============================================================

probability = logistic_model.predict_proba(
    new_student_scaled
)

print("\n========== PREDICTION PROBABILITY ==========")

print("Fail Probability:", probability[0][0])
print("Pass Probability:", probability[0][1])


# ============================================================
# 17. MODEL SAVE
# ============================================================

import joblib

joblib.dump(
    logistic_model,
    "student_pass_model.pkl"
)

joblib.dump(
    scaler,
    "student_scaler.pkl"
)

print("\nModel saved successfully!")


# ============================================================
# 18. MODEL LOAD
# ============================================================

loaded_model = joblib.load(
    "student_pass_model.pkl"
)

loaded_scaler = joblib.load(
    "student_scaler.pkl"
)

loaded_student = loaded_scaler.transform(
    new_student
)

loaded_prediction = loaded_model.predict(
    loaded_student
)

print("\n========== LOADED MODEL ==========")

if loaded_prediction[0] == 1:
    print("Loaded Model Prediction: PASS")
else:
    print("Loaded Model Prediction: FAIL")


# ============================================================
# 19. FEATURE IMPORTANCE - RANDOM FOREST
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": forest_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")
print(feature_importance)


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n================================================")
print("       SCIKIT-LEARN CONCEPTS COMPLETED")
print("================================================")

print("""
1. Dataset
2. Features (X)
3. Target (y)
4. LabelEncoder
5. Train-Test Split
6. StandardScaler
7. Logistic Regression
8. Linear Regression
9. Decision Tree
10. Random Forest
11. KNN
12. fit()
13. predict()
14. predict_proba()
15. Accuracy
16. Confusion Matrix
17. Classification Report
18. MAE
19. MSE
20. RMSE
21. R2 Score
22. Model Saving
23. Model Loading
24. Feature Importance
""")

print("All important Scikit-learn concepts completed!")