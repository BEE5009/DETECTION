# 📖 ดัชนี (Index) เอกสารทั้งหมด

หน้านี้เป็นดัชนีสำหรับการนำทางเอกสารการสอนทั้งหมดในโครงการ

---

## 🎯 เริ่มต้นจากที่นี่

### 1️⃣ **สรุปรวม (Overview)**
👉 [COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md)
- ภาพรวมของโครงการ
- โครงสร้างไฟล์และความสัมพันธ์
- ขั้นตอนการใช้งานทำเป็นลำดับ
- ตารางสรุปฟังก์ชันตามหมวดหมู่

---

## 📚 เอกสารตามไฟล์

### **hand_detection.py** (โปรแกรมหลัก)
👉 [FUNCTION_DOCUMENTATION.md](FUNCTION_DOCUMENTATION.md)

**ฟังก์ชันหลัก:**
- `_draw_unicode_text()` - วาดข้อความไทย
- `classify_gesture()` - จำแนกท่ามือ
- `run_with_solutions()` - วนลูปหลัก (Solutions API)
- `run_with_tasks()` - วนลูปหลัก (Tasks API)
- `run_on_images()` - รันบนรูปภาพ
- `main()` - จุดเข้าหลัก
- และอื่นๆ อีก 11 ฟังก์ชัน

**ปุ่มควบคุม:** Q, B, C, R, E, T, P

---

### **collect_imgs.py** (เก็บข้อมูล)
👉 [COLLECT_IMGS_DOCUMENTATION.md](COLLECT_IMGS_DOCUMENTATION.md)

**ฟังก์ชัน:**
- `adjust_brightness_contrast()` - ปรับความสว่าง/contrast
- `enhance_image_visibility()` - ปรับปรุงความชัด (CLAHE)
- `draw_countdown()` - วาด countdown
- `collect_samples()` - จับภาพ
- `wait_for_ready()` - รอให้พร้อม
- `main()` - จับภาพทั้ง 26 ตัวอักษร

**ผลลัพธ์:** 2,600 รูปสำหรับการฝึกสอน

---

### **create_dataset.py** (สร้างข้อมูล)
👉 [CREATE_DATASET_DOCUMENTATION.md](CREATE_DATASET_DOCUMENTATION.md)

**ขั้นตอน:**
1. โหลดภาพจาก `./data/`
2. ตรวจจับมือด้วย MediaPipe
3. สร้างเวกเตอร์ 42 มิติ
4. บันทึก `data.pickle`

---

### **train_classifier.py** (ฝึกสอน)
👉 [TRAIN_CLASSIFIER_DOCUMENTATION.md](TRAIN_CLASSIFIER_DOCUMENTATION.md)

**ขั้นตอน:**
1. โหลด `data.pickle`
2. แบ่ง train/test (80/20)
3. ฝึกสอน Random Forest
4. บันทึก `model.p`

---

### **test_mediapipe.py & verify_env.py** (ทดสอบ)
👉 [TEST_AND_VERIFY_DOCUMENTATION.md](TEST_AND_VERIFY_DOCUMENTATION.md)

**ฟังก์ชัน:**
- `test_mediapipe.py` - ตรวจสอบ MediaPipe
- `verify_env.py` - ตรวจสอบสภาพแวดล้อม
  - `show()` - แสดงข้อมูล package
  - `main()` - ตรวจสอบทั้งหมด

---

## 🔍 ค้นหาตามหมวดหมู่

### 📸 **Data Collection (เก็บข้อมูล)**
ดู: [COLLECT_IMGS_DOCUMENTATION.md](COLLECT_IMGS_DOCUMENTATION.md)

**ฟังก์ชันที่เกี่ยวข้อง:**
- `collect_samples()` - จับภาพ
- `wait_for_ready()` - รอยืนยัน
- `adjust_brightness_contrast()` - ปรับภาพ
- `enhance_image_visibility()` - ปรับความชัด

---

### 📊 **Data Processing (ประมวลผลข้อมูล)**
ดู: [CREATE_DATASET_DOCUMENTATION.md](CREATE_DATASET_DOCUMENTATION.md)

**ขั้นตอน:**
1. โหลดรูปภาพ
2. ตรวจจับจุดสำคัญมือ
3. สร้างเวกเตอร์

---

### 🤖 **Model Training (ฝึกสอนโมเดล)**
ดู: [TRAIN_CLASSIFIER_DOCUMENTATION.md](TRAIN_CLASSIFIER_DOCUMENTATION.md)

**ขั้นตอน:**
1. แบ่ง train/test
2. ฝึกสอน Random Forest
3. ประเมินและบันทึก

---

### 🎮 **Application (ใช้งาน)**
ดู: [FUNCTION_DOCUMENTATION.md](FUNCTION_DOCUMENTATION.md)

**ฟังก์ชันหลัก:**
- `run_with_solutions()` - โหมดกล้อง
- `run_with_tasks()` - โหมดกล้อง (Tasks)
- `run_on_images()` - โหมดรูปภาพ
- `classify_gesture()` - จำแนกท่า

---

### 🔍 **Diagnostics (วินิจฉัย)**
ดู: [TEST_AND_VERIFY_DOCUMENTATION.md](TEST_AND_VERIFY_DOCUMENTATION.md)

**เครื่องมือ:**
- `test_mediapipe.py` - ทดสอบ MediaPipe
- `verify_env.py` - ตรวจสอบ environments

---

## 🚀 ใช้งานอย่างไร

### ขั้นแรก: ตรวจสอบระบบ
```bash
python verify_env.py
```
👉 ดู: [TEST_AND_VERIFY_DOCUMENTATION.md](TEST_AND_VERIFY_DOCUMENTATION.md#-verify_envpy)

### ขั้นที่สอง: เก็บข้อมูล
```bash
python collect_imgs.py
```
👉 ดู: [COLLECT_IMGS_DOCUMENTATION.md](COLLECT_IMGS_DOCUMENTATION.md)

### ขั้นที่สาม: สร้างข้อมูลฝึกสอน
```bash
python create_dataset.py
```
👉 ดู: [CREATE_DATASET_DOCUMENTATION.md](CREATE_DATASET_DOCUMENTATION.md)

### ขั้นที่สี่: ฝึกสอนโมเดล
```bash
python train_classifier.py
```
👉 ดู: [TRAIN_CLASSIFIER_DOCUMENTATION.md](TRAIN_CLASSIFIER_DOCUMENTATION.md)

### ขั้นที่ห้า: ใช้งาน
```bash
python hand_detection.py
```
👉 ดู: [FUNCTION_DOCUMENTATION.md](FUNCTION_DOCUMENTATION.md)

---

## 📋 ตารางสรุปไฟล์

| ไฟล์ | ประเภท | วัตถุประสงค์ | เอกสาร |
|-----|-------|-----------|--------|
| `collect_imgs.py` | Data Collection | เก็บข้อมูล 2,600 รูป | [📖](COLLECT_IMGS_DOCUMENTATION.md) |
| `create_dataset.py` | Data Processing | สร้าง data.pickle | [📖](CREATE_DATASET_DOCUMENTATION.md) |
| `train_classifier.py` | Model Training | ฝึกสอนโมเดล | [📖](TRAIN_CLASSIFIER_DOCUMENTATION.md) |
| `hand_detection.py` | Main Application | ตรวจจับและจำแนก | [📖](FUNCTION_DOCUMENTATION.md) |
| `test_mediapipe.py` | Testing | ทดสอบ MediaPipe | [📖](TEST_AND_VERIFY_DOCUMENTATION.md#-test_mediapipepy) |
| `verify_env.py` | Diagnostics | ตรวจสอบสภาพแวดล้อม | [📖](TEST_AND_VERIFY_DOCUMENTATION.md#-verify_envpy) |

---

## 🎯 ค้นหาจากชื่อฟังก์ชัน

### A-C
- `adjust_brightness_contrast()` → [COLLECT_IMGS](COLLECT_IMGS_DOCUMENTATION.md)
- `classify_gesture()` → [FUNCTION](FUNCTION_DOCUMENTATION.md)
- `clear_banmai_template()` → [FUNCTION](FUNCTION_DOCUMENTATION.md)
- `collect_samples()` → [COLLECT_IMGS](COLLECT_IMGS_DOCUMENTATION.md)

### D
- `download_model()` → [FUNCTION](FUNCTION_DOCUMENTATION.md)
- `draw_countdown()` → [COLLECT_IMGS](COLLECT_IMGS_DOCUMENTATION.md)
- `draw_collection_status()` → [COLLECT_IMGS](COLLECT_IMGS_DOCUMENTATION.md)
- `draw_controls_info()` → [COLLECT_IMGS](COLLECT_IMGS_DOCUMENTATION.md)

### E-H
- `enhance_image_visibility()` → [COLLECT_IMGS](COLLECT_IMGS_DOCUMENTATION.md)
- `_find_thai_font_path()` → [FUNCTION](FUNCTION_DOCUMENTATION.md)

### I-L
- `is_banmai_pose()` → [FUNCTION](FUNCTION_DOCUMENTATION.md)
- `_landmark_distance()` → [FUNCTION](FUNCTION_DOCUMENTATION.md)
- `_list_image_files()` → [FUNCTION](FUNCTION_DOCUMENTATION.md)

### M-O
- `main()` → [COLLECT_IMGS](COLLECT_IMGS_DOCUMENTATION.md) / [FUNCTION](FUNCTION_DOCUMENTATION.md)
- `open_capture()` → [FUNCTION](FUNCTION_DOCUMENTATION.md)

### R
- `run_on_images()` → [FUNCTION](FUNCTION_DOCUMENTATION.md)
- `run_with_solutions()` → [FUNCTION](FUNCTION_DOCUMENTATION.md)
- `run_with_tasks()` → [FUNCTION](FUNCTION_DOCUMENTATION.md)

### S-T
- `save_banmai_template()` → [FUNCTION](FUNCTION_DOCUMENTATION.md)
- `show()` → [TEST_AND_VERIFY](TEST_AND_VERIFY_DOCUMENTATION.md)
- `test_mode()` → [FUNCTION](FUNCTION_DOCUMENTATION.md)

### W
- `wait_for_ready()` → [COLLECT_IMGS](COLLECT_IMGS_DOCUMENTATION.md)

---

## 📊 ความสัมพันธ์ของเอกสาร

```
COMPLETE_DOCUMENTATION.md (ศูนย์กลาง)
    │
    ├─→ FUNCTION_DOCUMENTATION.md (hand_detection.py)
    ├─→ COLLECT_IMGS_DOCUMENTATION.md (collect_imgs.py)
    ├─→ CREATE_DATASET_DOCUMENTATION.md (create_dataset.py)
    ├─→ TRAIN_CLASSIFIER_DOCUMENTATION.md (train_classifier.py)
    └─→ TEST_AND_VERIFY_DOCUMENTATION.md (test_* files)
```

---

## 🆘 Troubleshooting

### ปัญหา: ไม่รู้ว่าจะเริ่มจากอะไร
**วิธีแก้:** อ่าน [COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md)

### ปัญหา: หาฟังก์ชันที่ต้องการ
**วิธีแก้:** ใช้ "ค้นหาจากชื่อฟังก์ชัน" ด้านบน

### ปัญหา: เข้าใจวิธีข่ายรหัส
**วิธีแก้:** อ่านไฟล์ที่เกี่ยวข้องตามประเภท

### ปัญหา: เกิดข้อผิดพลาด
**วิธีแก้:** ดู "Troubleshooting" ในแต่ละเอกสาร

---

## 📲 เคล็ดลับการใช้เอกสาร

### ใน VS Code
1. ขยาย Explorer
2. ค้นหาไฟล์ `.md`
3. คลิกเพื่ออ่าน
4. Ctrl+F เพื่อค้นหา

### ใน GitHub
1. ใช้ Ctrl+F ค้นหา
2. คลิกลิงก์ในหัวข้อ
3. กลับที่ผ่านมาด้วย Alt+Left

### บนเครื่อง
```bash
# ค้นหาฟังก์ชัน
grep -r "def function_name" *.md

# อ่านไฟล์
less FUNCTION_DOCUMENTATION.md
```

---

## 🎓 เรียนรู้ลำดับที่แนะนำ

1. **ผู้เริ่มต้น:**
   - อ่าน [COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md)
   - ตามขั้นตอน "ขั้นตอนการใช้งานทั้งครั้ง"

2. **นักพัฒนา:**
   - อ่านเอกสารตามไฟล์ในลำดับของการใช้

3. **บำรุงรักษา:**
   - ดูตารางความสัมพันธ์ของไฟล์
   - ศึกษาแต่ละฟังก์ชัน

---

## 📝 รายการอ្នុក្រម

### ไฟล์เอกสาร
- ✅ `COMPLETE_DOCUMENTATION.md` - สรุปรวม
- ✅ `FUNCTION_DOCUMENTATION.md` - hand_detection.py
- ✅ `COLLECT_IMGS_DOCUMENTATION.md` - collect_imgs.py
- ✅ `CREATE_DATASET_DOCUMENTATION.md` - create_dataset.py
- ✅ `TRAIN_CLASSIFIER_DOCUMENTATION.md` - train_classifier.py
- ✅ `TEST_AND_VERIFY_DOCUMENTATION.md` - test files
- ✅ `INDEX.md` - หน้านี้

### ไฟล์โปรแกรม
- ✅ `hand_detection.py` - โปรแกรมหลัก (18 ฟังก์ชัน)
- ✅ `collect_imgs.py` - เก็บข้อมูล (8 ฟังก์ชัน)
- ✅ `create_dataset.py` - สร้างข้อมูล
- ✅ `train_classifier.py` - ฝึกสอน
- ✅ `test_mediapipe.py` - ทดสอบ
- ✅ `verify_env.py` - ตรวจสอบ (2 ฟังก์ชัน)

---

## 🔗 ลิงก์ด่วน

| เอกสาร | ลิงก์ | ไฟล์ที่เกี่ยวข้อง |
|--------|------|-----------------|
| สรุปรวม | [📖](COMPLETE_DOCUMENTATION.md) | ทั้งหมด |
| ฟังก์ชัน | [📖](FUNCTION_DOCUMENTATION.md) | hand_detection.py |
| เก็บข้อมูล | [📖](COLLECT_IMGS_DOCUMENTATION.md) | collect_imgs.py |
| สร้างข้อมูล | [📖](CREATE_DATASET_DOCUMENTATION.md) | create_dataset.py |
| ฝึกสอน | [📖](TRAIN_CLASSIFIER_DOCUMENTATION.md) | train_classifier.py |
| ทดสอบ | [📖](TEST_AND_VERIFY_DOCUMENTATION.md) | test_mediapipe.py, verify_env.py |

---

## 🎉 เสร็จสิ้น!

ตอนนี้คุณมีเอกสารที่ครอบคลุมทั้งหมด:
- ✅ สรุปรวมโครงการ
- ✅ รายละเอียดแต่ละไฟล์
- ✅ คำอธิบายแต่ละฟังก์ชัน
- ✅ ขั้นตอนการใช้งาน
- ✅ แนวทางการแก้ไขปัญหา
- ✅ ดัชนีการนำทาง

**เริ่มจาก [COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md) 👈**
