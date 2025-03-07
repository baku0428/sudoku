import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from sklearn.model_selection import train_test_split

# 🔹 1. โหลดข้อมูลซูโดกุจาก CSV
df = pd.read_csv("sudoku.csv")

# แปลงปริศนาและคำตอบให้เป็นอาร์เรย์ของตัวเลข
X = np.array([list(map(int, list(puzzle))) for puzzle in df["quizzes"]])
y = np.array([list(map(int, list(solution))) for solution in df["solutions"]])

# แบ่งข้อมูลเป็น train และ test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# ปรับขนาดให้เข้ากับโมเดล
X_train = X_train.reshape(-1, 9, 9, 1)
X_test = X_test.reshape(-1, 9, 9, 1)

# 🔹 2. แปลง y_train และ y_test ให้เป็น One-Hot
num_classes = 10  # เพราะตัวเลขใน Sudoku คือ 0-9
y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes).reshape(-1, 9, 9, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes).reshape(-1, 9, 9, num_classes)

# 🔹 3. สร้างโมเดล Deep Learning
model = Sequential([
    Flatten(input_shape=(9, 9, 1)),  # แปลงเป็นเวกเตอร์
    Dense(128, activation="relu"),
    Dense(256, activation="relu"),
    Dense(512, activation="relu"),
    Dense(9 * 9 * num_classes, activation="softmax"),  # ทำนายค่าทั้ง 81 ช่อง (one-hot)
    Reshape((9, 9, num_classes))  # กลับเป็นรูปแบบ 9x9x10
])

# 🔹 4. คอมไพล์และฝึกโมเดล
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 🔹 5. ฟังก์ชันทำนายซูโดกุ
def solve_sudoku(puzzle):
    puzzle = np.array([list(map(int, list(puzzle)))])
    puzzle = puzzle.reshape(-1, 9, 9, 1)
    prediction = model.predict(puzzle)
    return np.argmax(prediction, axis=-1).reshape(9, 9)  # แปลงจาก one-hot กลับเป็นตัวเลข

# 🔹 6. ทดสอบโมเดลกับตัวอย่างปริศนา
test_puzzle = df["quizzes"][0]
solved_board = solve_sudoku(test_puzzle)
print("🔹 ปริศนา Sudoku ที่ให้บอทแก้:\n", np.array(list(test_puzzle)).reshape(9, 9))
print("\n✅ คำตอบที่บอทแก้ได้:\n", solved_board)
