import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from sklearn.model_selection import train_test_split

# 🔹 โหลดข้อมูล Sudoku
df = pd.read_csv("sudoku.csv")

# แปลงปริศนาและคำตอบให้เป็นอาร์เรย์
X = np.array([list(map(int, list(p))) for p in df["quizzes"]])
y = np.array([list(map(int, list(s))) for s in df["solutions"]])

# แบ่งข้อมูล train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# รีเชปข้อมูล
X_train = X_train.reshape(-1, 9, 9, 1)
X_test = X_test.reshape(-1, 9, 9, 1)

# แปลง `y_train` เป็น One-Hot
num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes).reshape(-1, 9, 9, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes).reshape(-1, 9, 9, num_classes)

# 🔹 สร้างโมเดล
model = Sequential([
    Flatten(input_shape=(9, 9, 1)),
    Dense(128, activation="relu"),
    Dense(256, activation="relu"),
    Dense(512, activation="relu"),
    Dense(9 * 9 * num_classes, activation="softmax"),
    Reshape((9, 9, num_classes))
])

# คอมไพล์และเทรน
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 🔹 บันทึกโมเดล
model.save("sudoku_model.h5")
print("✅ โมเดลถูกบันทึกสำเร็จ!")
