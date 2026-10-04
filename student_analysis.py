from pathlib import Path

import pandas as pd


# ==========================================
# 1. READ DATASET
# ==========================================

DATASET_PATH = Path(__file__).resolve().parent / "student.csv"
df = pd.read_csv(DATASET_PATH)

print("\n===== COMPLETE DATASET =====")
print(df)


# ==========================================
# 2. FIRST 5 ROWS
# ==========================================

print("\n===== FIRST 5 STUDENTS =====")
print(df.head())


# ==========================================
# 3. LAST 5 ROWS
# ==========================================

print("\n===== LAST 5 STUDENTS =====")
print(df.tail())


# ==========================================
# 4. DATASET SHAPE
# ==========================================

print("\n===== DATASET SHAPE =====")
print("Rows and Columns:", df.shape)


# ==========================================
# 5. DATASET INFORMATION
# ==========================================

print("\n===== DATASET INFORMATION =====")
df.info()


# ==========================================
# 6. STATISTICAL SUMMARY
# ==========================================

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())


# ==========================================
# 7. COLUMN NAMES
# ==========================================

print("\n===== COLUMN NAMES =====")
print(df.columns)


# ==========================================
# 8. DATA TYPES
# ==========================================

print("\n===== DATA TYPES =====")
print(df.dtypes)


# ==========================================
# 9. SELECT ONE COLUMN
# ==========================================

print("\n===== STUDENT NAMES =====")
print(df["Name"])


# ==========================================
# 10. SELECT MULTIPLE COLUMNS
# ==========================================

print("\n===== NAME AND MARKS =====")
print(df[["Name", "Maths", "Physics", "Python", "ML"]])


# ==========================================
# 11. SELECT ROW USING iloc
# ==========================================

print("\n===== FIRST STUDENT =====")
print(df.iloc[0])


# ==========================================
# 12. SELECT SPECIFIC VALUE
# ==========================================

print("\n===== FIRST STUDENT MATHS MARKS =====")
print(df.iloc[0]["Maths"])


# ==========================================
# 13. FILTER STUDENTS
# ==========================================

print("\n===== STUDENTS WITH MARKS > 80 IN MATHS =====")
print(df[df["Maths"] > 80])


# ==========================================
# 14. FILTER USING MULTIPLE CONDITIONS
# ==========================================

print("\n===== STUDENTS WITH ATTENDANCE > 85 AND STUDY HOURS > 4 =====")

filtered_students = df[
    (df["Attendance"] > 85) &
    (df["Study_Hours"] > 4)
]

print(filtered_students)


# ==========================================
# 15. SORT BY MARKS
# ==========================================

print("\n===== STUDENTS SORTED BY ML MARKS =====")

sorted_students = df.sort_values(
    "ML",
    ascending=False
)

print(sorted_students[["Name", "ML"]])


# ==========================================
# 16. CREATE TOTAL MARKS
# ==========================================

subjects = [
    "Maths",
    "Physics",
    "Chemistry",
    "Python",
    "ML"
]

df["Total_Marks"] = df[subjects].sum(axis=1)

print("\n===== TOTAL MARKS =====")
print(df[["Name", "Total_Marks"]])


# ==========================================
# 17. CREATE AVERAGE MARKS
# ==========================================

df["Average_Marks"] = df[subjects].mean(axis=1)

print("\n===== AVERAGE MARKS =====")
print(df[["Name", "Average_Marks"]])


# ==========================================
# 18. ROUND AVERAGE
# ==========================================

df["Average_Marks"] = df["Average_Marks"].round(2)

print("\n===== ROUNDED AVERAGE =====")
print(df[["Name", "Average_Marks"]])


# ==========================================
# 19. PASS / FAIL
# ==========================================

df["Result"] = df["Average_Marks"] >= 40

print("\n===== PASS / FAIL =====")
print(df[["Name", "Average_Marks", "Result"]])


# ==========================================
# 20. CHANGE TRUE/FALSE TO PASS/FAIL
# ==========================================

df["Result"] = df["Result"].map({
    True: "PASS",
    False: "FAIL"
})

print("\n===== FINAL RESULT =====")
print(df[["Name", "Average_Marks", "Result"]])


# ==========================================
# 21. HIGHEST MARKS
# ==========================================

print("\n===== HIGHEST MARKS =====")

print("Highest Maths:", df["Maths"].max())
print("Highest Physics:", df["Physics"].max())
print("Highest Chemistry:", df["Chemistry"].max())
print("Highest Python:", df["Python"].max())
print("Highest ML:", df["ML"].max())


# ==========================================
# 22. LOWEST MARKS
# ==========================================

print("\n===== LOWEST MARKS =====")

print("Lowest Maths:", df["Maths"].min())
print("Lowest Physics:", df["Physics"].min())
print("Lowest Chemistry:", df["Chemistry"].min())
print("Lowest Python:", df["Python"].min())
print("Lowest ML:", df["ML"].min())


# ==========================================
# 23. SUBJECT-WISE AVERAGE
# ==========================================

print("\n===== SUBJECT-WISE AVERAGE =====")

subject_average = df[subjects].mean()

print(subject_average)


# ==========================================
# 24. TOPPER
# ==========================================

topper_index = df["Total_Marks"].idxmax()

print("\n===== TOPPER =====")

print("Name:", df.loc[topper_index, "Name"])
print("Total Marks:", df.loc[topper_index, "Total_Marks"])
print("Average:", df.loc[topper_index, "Average_Marks"])


# ==========================================
# 25. LOWEST PERFORMER
# ==========================================

lowest_index = df["Total_Marks"].idxmin()

print("\n===== LOWEST PERFORMER =====")

print("Name:", df.loc[lowest_index, "Name"])
print("Total Marks:", df.loc[lowest_index, "Total_Marks"])
print("Average:", df.loc[lowest_index, "Average_Marks"])


# ==========================================
# 26. GENDER COUNT
# ==========================================

print("\n===== GENDER COUNT =====")

print(df["Gender"].value_counts())


# ==========================================
# 27. GENDER-WISE AVERAGE
# ==========================================

print("\n===== GENDER-WISE AVERAGE =====")

gender_average = df.groupby("Gender")["Average_Marks"].mean()

print(gender_average)


# ==========================================
# 28. STUDY HOURS ANALYSIS
# ==========================================

print("\n===== STUDY HOURS ANALYSIS =====")

print("Maximum Study Hours:",
      df["Study_Hours"].max())

print("Minimum Study Hours:",
      df["Study_Hours"].min())

print("Average Study Hours:",
      df["Study_Hours"].mean())


# ==========================================
# 29. ATTENDANCE ANALYSIS
# ==========================================

print("\n===== ATTENDANCE ANALYSIS =====")

print("Maximum Attendance:",
      df["Attendance"].max())

print("Minimum Attendance:",
      df["Attendance"].min())

print("Average Attendance:",
      df["Attendance"].mean())


# ==========================================
# 30. HIGH ATTENDANCE STUDENTS
# ==========================================

print("\n===== HIGH ATTENDANCE STUDENTS =====")

high_attendance = df[
    df["Attendance"] >= 90
]

print(
    high_attendance[
        ["Name", "Attendance", "Average_Marks"]
    ]
)


# ==========================================
# 31. HIGH STUDY HOURS STUDENTS
# ==========================================

print("\n===== STUDENTS STUDYING MORE THAN 4 HOURS =====")

high_study = df[
    df["Study_Hours"] > 4
]

print(
    high_study[
        ["Name", "Study_Hours", "Average_Marks"]
    ]
)


# ==========================================
# 32. DUPLICATE CHECK
# ==========================================

print("\n===== DUPLICATE CHECK =====")

print(df.duplicated().sum())


# ==========================================
# 33. MISSING VALUE CHECK
# ==========================================

print("\n===== MISSING VALUES =====")

print(df.isnull().sum())


# ==========================================
# 34. CORRELATION
# ==========================================

print("\n===== CORRELATION =====")

print(
    df[
        ["Study_Hours", "Attendance", "Average_Marks"]
    ].corr()
)


# ==========================================
# 35. FINAL STUDENT REPORT
# ==========================================

print("\n========================================")
print("       FINAL STUDENT REPORT")
print("========================================")

final_report = df[
    [
        "Student_ID",
        "Name",
        "Gender",
        "Study_Hours",
        "Attendance",
        "Total_Marks",
        "Average_Marks",
        "Result"
    ]
]

print(final_report)


# ==========================================
# 36. SAVE FINAL REPORT
# ==========================================

final_report.to_csv(
    "final_student_report.csv",
    index=False
)

print("\nFinal report saved successfully!")