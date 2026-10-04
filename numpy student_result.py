import numpy as np

names = np.array([
    "Amit",
    "Priya",
    "Rahul",
    "Sneha",
    "Pooja",
    "Akash",
    "Neha",
    "Rohit"
])

subjects = np.array([
    "Maths",
    "Physics",
    "Chemistry",
    "Python",
    "ML"
])

marks = np.array([
    [78, 85, 92, 88, 76],
    [65, 72, 68, 70, 75],
    [90, 95, 88, 92, 96],
    [55, 60, 58, 65, 62],
    [82, 78, 85, 80, 88],
    [70, 75, 72, 68, 74],
    [95, 92, 96, 94, 90],
    [60, 65, 70, 62, 68]
])

print(marks)
print("Shape:", marks.shape)
print("Dimensions:", marks.ndim)
print("Data Type:", marks.dtype)
print("Amit marks:", marks[0])
print("Amit Python:", marks[0, 3])
total_marks = np.sum(marks, axis=1)

print("Total Marks:")
print(total_marks)
average_marks = np.mean(marks, axis=1)
#average marks
print("Average Marks:")
print(average_marks)
#median marks
median_marks = np.median(marks, axis=1)

print("Median Marks:")
print(median_marks)
#highest marks
print("Highest Mark:", np.max(marks))
print("Lowest Mark:", np.min(marks))
print("Class Average:", np.mean(marks))
print("Standard Deviation:", np.std(marks))
subject_average = np.mean(marks, axis=0)

print("Subject-wise Average:")
print(subject_average)
topper_index = np.argmax(total_marks)

print("Topper:", names[topper_index])
print("Topper Total:", total_marks[topper_index])
lowest_index = np.argmin(total_marks)

print("Lowest Student:", names[lowest_index])
print("Lowest Total:", total_marks[lowest_index])
passed = np.all(marks >= 40, axis=1)

print("Pass/Fail:")
print(passed)
print("\n===== STUDENT REPORT =====")

for i in range(len(names)):

    print("\nName:", names[i])
    print("Total:", total_marks[i])
    print("Average:", round(average_marks[i], 2))
    print("Median:", median_marks[i])

    if passed[i]:
        print("Result: PASS")
    else:
        print("Result: FAIL")