from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# โหลดโมเดลที่ฝึกไว้
model = tf.keras.models.load_model("sudoku_model.h5")

# ✅ ตรวจสอบว่า num สามารถใส่ในตำแหน่ง (row, col) ได้หรือไม่
def is_valid(board, row, col, num):
    for i in range(9):
        if board[row][i] == num or board[i][col] == num:
            return False

    start_row, start_col = (row // 3) * 3, (col // 3) * 3
    for i in range(3):
        for j in range(3):
            if board[start_row + i][start_col + j] == num:
                return False
    return True

# ✅ ฟังก์ชันแก้ Sudoku ด้วย Backtracking
def solve_sudoku(board):
    empty_cell = find_empty_cell(board)
    if not empty_cell:
        return True

    row, col = empty_cell
    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row][col] = num
            if solve_sudoku(board):
                return True
            board[row][col] = 0  # Undo ถ้าใส่เลขนี้แล้วแก้ไม่ได้
    return False

# ✅ ค้นหาช่องว่างที่ยังไม่มีตัวเลข
def find_empty_cell(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return (i, j)
    return None

# ✅ ใช้โมเดล AI คาดเดา
def predict_sudoku(puzzle):
    puzzle = np.array(puzzle).reshape(1, 9, 9, 1)  # ปรับรูปแบบให้ตรงกับโมเดล
    prediction = model.predict(puzzle)
    predicted_board = np.argmax(prediction, axis=-1).reshape(9, 9)
    return predicted_board

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    data = request.get_json()
    puzzle = np.array(data['puzzle']).reshape(9, 9)

    if puzzle.shape != (9, 9):
        return jsonify({'error': 'Invalid Sudoku grid size'}), 400

    # ใช้โมเดล AI คาดเดา
    ai_solution = predict_sudoku(puzzle)

    # ตรวจสอบความถูกต้อง
    if all(is_valid(ai_solution, r, c, ai_solution[r][c]) for r in range(9) for c in range(9) if puzzle[r][c] == 0):
        return jsonify({'solution': ai_solution.tolist()})
    else:
        # ถ้า AI ผิด ให้ใช้ Backtracking แทน
        if solve_sudoku(puzzle):
            return jsonify({'solution': puzzle.tolist()})
        else:
            return jsonify({'error': 'No solution found'}), 400

if __name__ == '__main__':
    app.run(debug=True)
