# sudoku
241-202 miniproject

# อธิบายขั้นตอนการติดตั้ง

โครงการนี้เป็นโปรเจกต์แก้ปริศนา Sudoku โดยใช้ Machine Learning ด้วยโมเดล TensorFlow และมี Flask เป็น Backend เพื่อให้สามารถป้อนข้อมูล Sudoku ผ่านหน้าเว็บและให้โมเดลช่วยแก้ปริศนาให้โดยอัตโนมัติ

---

### 1 ติดตั้ง Python และสร้าง Virtual Environment

```bash
# ติดตั้ง Python (ถ้ายังไม่มี)
# ดาวน์โหลดได้จาก https://www.python.org/downloads/

# สร้าง Virtual Environment
python -m venv venv

# เปิดใช้งาน Virtual Environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 2️ ติดตั้ง Dependencies ที่จำเป็น

```bash
pip install -r requirements.txt
```

หรือหากไม่มี `requirements.txt` ให้ติดตั้งแพ็กเกจที่จำเป็นด้วยคำสั่งนี้:

```bash
pip install flask numpy pandas tensorflow scikit-learn
```

### 3️ เตรียมไฟล์ข้อมูล

ตรวจสอบว่าคุณมีไฟล์ที่จำเป็นดังนี้:
- `app.py` → ไฟล์หลักสำหรับรัน Flask Server
- `templates/index.html` → ไฟล์ HTML สำหรับแสดงผล UI
- `sudoku_model.h5` → ไฟล์โมเดลที่ถูกฝึกมาแล้ว
- `train_model.py` → ไฟล์สำหรับฝึกโมเดลใหม่ (หากต้องการ)
- `sudoku.csv` → ไฟล์ข้อมูล Sudoku สำหรับการฝึกโมเดล

---

## วิธีใช้งาน

  รันเว็บแอปพลิเคชัน

```bash
python app.py
```

จากนั้นเปิดเบราว์เซอร์และไปที่:
```
http://127.0.0.1:5000
```

  การฝึกโมเดลใหม่ (ถ้าต้องการ)

หากต้องการฝึกโมเดลใหม่จาก `sudoku.csv` สามารถใช้ไฟล์ `train_model.py` ได้:

```bash
python train_model.py
```

โมเดลใหม่จะถูกบันทึกเป็น `sudoku_model.h5`

---

##  โครงสร้างไฟล์

```
📂 Sudoku
├── app.py               # ไฟล์ Flask Backend
├── train_model.py       # ไฟล์ฝึกโมเดลใหม่
├── sudoku_model.h5      # โมเดลที่ถูกฝึกแล้ว
├── sudoku.csv           # ข้อมูลปริศนา Sudoku
├── templates/
│   ├── index.html       # UI ของแอปพลิเคชัน
└── README.md            # คำแนะนำการใช้งาน
```

---

##  หมายเหตุ
- โมเดล `sudoku_model.h5` ต้องอยู่ในไดเรกทอรีเดียวกับ `app.py`
- หากรัน `app.py` แล้วหน้าเว็บไม่แสดงผล ให้ตรวจสอบว่าติดตั้ง Flask แล้วหรือไม่
- หากต้องการฝึกโมเดลใหม่ ให้แน่ใจว่ามีไฟล์ `sudoku.csv` และติดตั้ง TensorFlow
- โหลดไฟล์ `sudoku.csv` ได้ใน team เนื่องจากไฟล์มีขนาดใหญ่เกินไปไม่สามาถดาวโหลดลงมาใน git ได้
