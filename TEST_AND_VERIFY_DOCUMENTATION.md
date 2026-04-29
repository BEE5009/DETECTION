# 📄 เอกสารฟังก์ชัน - test_mediapipe.py & verify_env.py

เอกสารนี้บรรยายฟังก์ชันในสคริปต์ทดสอบสองไฟล์

---

## 📋 test_mediapipe.py

### วัตถุประสงค์
ตรวจสอบว่า MediaPipe ติดตั้งแล้วหรือไม่ และมี API ไหนพร้อมใช้

### ขั้นตอนการทำงาน

1. **โหลด MediaPipe**
   ```python
   import mediapipe as mp
   ```

2. **ตรวจสอบเวอร์ชัน**
   ```python
   print(f"mediapipe version: {mp.__version__}")
   ```

3. **ตรวจสอบ Solutions API**
   ```python
   print(f"has solutions: {hasattr(mp, 'solutions')}")
   ```

4. **ตรวจสอบ Hands Module** (ถ้า solutions พร้อม)
   ```python
   print(f"has hands: {hasattr(mp.solutions, 'hands')}")
   ```

### ผลลัพธ์ที่คาดหวัง

**ถ้า MediaPipe สมบูรณ์:**
```
mediapipe version: 0.8.10.1
has solutions: True
solutions found!
has hands: True
```

**ถ้า MediaPipe ไม่มี Solutions:**
```
mediapipe version: 0.8.10.1
has solutions: False
ERROR: mediapipe does not have 'solutions' attribute
Available attributes: ['Account', 'Graph', 'Image', ...]
```

### การใช้

```bash
python test_mediapipe.py
```

---

## 📋 verify_env.py

### วัตถุประสงค์
ตรวจสอบสภาพแวดล้อม Python: เวอร์ชัน package, เส้นทาง, import status

### ฟังก์ชัน

#### `show(name)`
**วัตถุประสงค์:** แสดงข้อมูล package เดียว

**พารามิเตอร์:**
- `name` - ชื่อของ package (เช่น 'numpy', 'cv2')

**ขั้นตอนการทำงาน:**
1. ลองโหลด module ด้วย `importlib.import_module`
2. ดึงเวอร์ชัน (`__version__`)
3. ดึงเส้นทางไฟล์ (`__file__`)
4. พิมพ์ผลลัพธ์

**ผลลัพธ์:**
```
numpy: version=1.24.0, file=/path/to/numpy/__init__.py
```

**หากเกิดข้อผิดพลาด:**
```
numpy: ERROR: [error message]
```

#### `main()`
**วัตถุประสงค์:** ฟังก์ชันหลักสำหรับการตรวจสอบสภาพแวดล้อมทั้งหมด

**ขั้นตอนการทำงาน:**

1. **แสดงข้อมูล Python**
   ```python
   print('PYTHON:', sys.executable)
   print('sys.path[0]:', sys.path[0])
   ```

2. **ตรวจสอบ Key Packages**
   - `numpy` - สำหรับการประมวลผลอาร์เรย์
   - `cv2` - OpenCV
   - MediaPipe
     - `__version__`
     - มี `solutions` API?
     - มี `tasks` API?

3. **ลองนำเข้า `hand_detection.py`**
   - ตรวจสอบว่า module สามารถนำเข้าได้หรือไม่
   - แสดงข้อความ "OK" หรือ "ERROR" พร้อม traceback

### ผลลัพธ์ตัวอย่าง

```
PYTHON: C:\path\to\.venv\Scripts\python.exe
sys.path[0]: .
numpy: version=1.24.0, file=C:\path\to\numpy\__init__.py
cv2: version=4.6.0, file=C:\path\to\cv2\__init__.py
mediapipe.__file__: C:\path\to\mediapipe\__init__.py
mediapipe has solutions: True
mediapipe has tasks: True
hand_detection import: OK
```

### การใช้

```bash
python verify_env.py
```

### เมื่อใดควรใช้

1. **วินิจฉัยปัญหา import**
   ```bash
   python verify_env.py
   ```

2. **ตรวจสอบ venv ทำงานถูกต้อง**
   - ตรวจสอบ `PYTHON` ชี้ไปยัง venv จริงๆ หรือไม่
   - ตรวจสอบ packages ติดตั้งในวิทยบ

3. **ก่อนรับ hand_detection.py**
   - รับรองว่า module สามารถนำเข้าได้
   - ดู traceback หากเกิดข้อผิดพลาด

---

## 🔍 เทียบเคียง: test_mediapipe.py vs verify_env.py

| ลักษณะ | test_mediapipe.py | verify_env.py |
|-------|------------------|--------------|
| **ขอบเขต** | MediaPipe เท่านั้น | ทั้ง environments + packages |
| **วัตถุประสงค์** | เร็ว ๆ check MediaPipe | เข้มข้นตรวจสอบทั้งหมด |
| **ความซับซ้อน** | ง่าย (10 บรรทัด) | ปานกลาง (40+ บรรทัด) |
| **ความใช้งาน** | ก่อนใช้ hand_detection | วินิจฉัยปัญหา |

---

## 🎯 Troubleshooting

### ปัญหา: `ModuleNotFoundError: No module named 'mediapipe'`

**วิธีแก้:**
```bash
pip install mediapipe
```

### ปัญหา: MediaPipe ติดตั้งแล้วแต่ไม่มี `solutions`

**พยายาม:**
1. อัพเกรด MediaPipe
   ```bash
   pip install --upgrade mediapipe
   ```
2. ถ้าไม่สำเร็จ ลอง tasks API แทน

### ปัญหา: `hand_detection import: ERROR`

**ดูผลลัพธ์:**
- ดู traceback ที่พิมพ์ออกมา
- ดู requirements ที่ขาดหายไป
- ลอง verify_env.py เพื่อดูรายละเอียด

---

## 💡 ข้อเสนอหนึ่ง

หากต้องการเช็คได้อย่างรวดเร็ว:

**สำหรับสมาชิก:**
```bash
python verify_env.py 2>&1 | grep -E "numpy|cv2|mediapipe|hand_detection"
```

**สำหรับ Windows:**
```powershell
python verify_env.py | findstr "numpy cv2 mediapipe hand_detection"
```

---

## 📝 หมายเหตุ

- `__version__` อาจไม่มีใน modules บางตัว → ดึง `None`
- `__file__` อาจเป็น `'builtin'` สำหรับ built-in modules
- เส้นทางอาจแตกต่างตามระบบปฏิบัติการและ venv
