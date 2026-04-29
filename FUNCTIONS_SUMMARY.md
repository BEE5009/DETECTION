# Hand Detection Project - Function Summary

## การเก็บ และจัดการข้อมูลไทย Unicode

### 1. **_draw_unicode_text(img, text, position, font_size=32, color=(0, 255, 0))**
   - วาดข้อความ Unicode (ตัวอักษรไทย) บน OpenCV image
   - Fallback ไปใช้ `cv2.putText()` ถ้า Pillow ไม่พร้อม
   - รองรับฟอนต์ไทยจากหลายแหล่ง

### 2. **_find_thai_font_path()** (nested in _draw_unicode_text)
   - ค้นหาไฟล์ฟอนต์ที่รองรับภาษาไทย
   - ค้นหาใน: local folders, system fonts, และตั้งค่าเริ่มต้น

## UI และภาษา

### 3. **get_ui_text(key)**
   - ดึงข้อความ UI ตามภาษาที่เลือก (TH/EN)
   - Return: String text สำหรับแสดงผล

### 4. **toggle_language()**
   - สลับภาษา UI ระหว่าง Thai (ไทย) และ English
   - Global variable `UI_LANGUAGE` ถูกอัปเดต

## โมเดลและการจำแนกท่า

### 5. **init_gesture_model()**
   - โหลดโมเดลท่ามือจาก `model.p` (ไฟล์ pickle)
   - ตั้งค่า global variables: `_gesture_model`, `_gesture_label_map`
   - พิมพ์การรับตรวจสอบลอง (debugging info)

### 6. **hand_landmarks_to_vector(landmarks)**
   - แปลงจุดแนวเขื่องมือ (hand landmarks) เป็น numpy vector
   - Normalize โดยใช้จุดต่ำสุด (min_x, min_y)
   - Return: numpy array shape (1, 42) หรือ None ถ้าข้อมูลไม่ถูกต้อง

### 7. **classify_gesture_model(hand_landmarks)**
   - จำแนกท่ามือโดยใช้โมเดล ML ที่ฝึกแล้ว
   - ใช้ `hand_landmarks_to_vector()` เพื่อสร้าง vector
   - Return: label string (เช่น 'A', 'B', ...) หรือ None

### 8. **classify_gesture(hand_landmarks)**
   - จำแนกท่ามือเป็นตัวอักษรไทย ก-ฮ โดยใช้ logic heuristic
   - ตรวจสอบว่านิ้วไหนขยาย (extended) โดยใช้จุด geometry
   - Return: ตัวอักษรไทย (Thai letter) หรือ "?" ถ้าไม่พบท่า

## Template Pose Matching

### 9. **_normalize_landmarks(landmarks)**
   - Normalize hand landmarks สำหรับการเปรียบเทียบท่า
   - ใช้ wrist (landmark 0) เป็น origin
   - Scale โดยใช้ระยะทางสูงสุด
   - Return: list of tuples (x, y, z) normalized

### 10. **_landmark_distance(a, b)**
   - คำนวณระยะทาง Euclidean เฉลี่ยระหว่าง landmark sets
   - Return: float (distancec) หรือ inf ถ้าข้อมูลไม่ตรงกัน

### 11. **is_banmai_pose(hand_landmarks, threshold=0.12)**
   - ตรวจสอบว่าท่าปัจจุบันตรงกับ template "บ้านลับ" ที่บันทึกไว้
   - ใช้ `_normalize_landmarks()` และ `_landmark_distance()`
   - Return: True/False

### 12. **save_banmai_template(hand_landmarks)**
   - บันทึกท่ามือปัจจุบันเป็น template "บ้านลับ"
   - ใช้ `_normalize_landmarks()` 
   - Return: True ถ้าบันทึกสำเร็จ

### 13. **clear_banmai_template()**
   - ลบ template "บ้านลับ" ที่บันทึกไว้
   - Set global `_BANMAI_TEMPLATE = None`

## โมเดลและการตั้งค่า

### 14. **download_model(url)**
   - ดาวน์โหลด MediaPipe model (.task file) จาก URL
   - Return: ไฟล์ path ของ temp file ที่ดาวน์โหลด
   - Raise: Exception ถ้าดาวน์โหลดล้มเหลว

### 15. **open_capture(camera_index=0, video_path=None)**
   - เปิดวิดีโอ capture จากกล้องเว็บแคม หรือ ไฟล์วิดีโอ
   - ลองหลายดัชนี (index) ของกล้องถ้าไม่พบ
   - Return: `cv2.VideoCapture` object

## การประมวลผลการหาท่า

### 16. **run_with_solutions(cap, max_num_hands, min_detection_confidence)**
   - ใช้ MediaPipe solutions API (mp.solutions.hands)
   - สตรมเรียลไทม์ จากกล้อง โดย capture, detect, classify
   - รายการสนับสนุน:
     - `A`: สลับภาษา
     - `B`: บันทึกท่า "บ้านลับ"
     - `C`: ล้างท่า
     - `R`: บันทึกตัวอักษร
     - `E`: ลบตัวอักษร
     - `T`: แสดงผลลัพธ์
     - `P`: พิมพ์ recognized words
     - `Q`: ออก

### 17. **run_with_tasks(cap, model_path, max_num_hands, min_detection_confidence)**
   - ใช้ MediaPipe Tasks API (HandLandmarker)
   - ประกอบการคำนวณวิดีโอสตรมเรียลไทม์
   - รองรับคีย์เดียวกับ `run_with_solutions()`

## การประมวลผลภาพ

### 18. **_list_image_files(dir_path)**
   - ค้นหา image files ทั้งหมดใน directory
   - รงรับ: jpg, jpeg, png, bmp, webp
   - Return: sorted list of file paths

### 19. **_read_image(path)**
   - อ่าน image จาก disk โดยรองรับ Unicode paths (Windows)
   - Fallback ไปใช้ Pillow ถ้า OpenCV ล้มเหลว
   - Return: numpy array (BGR format) หรือ None

### 20. **run_on_images(dir_path, max_num_hands, min_detection_confidence, model=None)**
   - จำแนกท่ามือจากรูปภาพทั้งหมดในโฟลเดอร์
   - เปรียบเทียบกับ filename label
   - บันทึกรูปผลลัพธ์ใน 'out' subdirectory
   - พิมพ์ accuracy report

## หลัก และทดสอบ

### 21. **main(camera_index=0, video_path=None, max_num_hands=2, min_detection_confidence=0.5, model=None, pic_dir=None)**
   - ฟังก์ชันหลักของโปรแกรม
   - ตัวเลือก:
     - ถ้า `pic_dir`: ใช้ `run_on_images()`
     - ถ้าไม่: ใช้กระโปรแกรมเกราะ (solutions/tasks)
   - Initialize gesture model
   - UTF-8 encoding setup สำหรับ console

### 22. **test_mode(max_num_hands=2, min_detection_confidence=0.5, model=None)**
   - ทดสอบว่า MediaPipe API ไหนพร้อม (solutions vs tasks)
   - ไม่ต้องใช้กล้อง
   - บริจากข้อมูล debug information

---

## Global Variables (ตัวแปรส่วนกลาง)

- `_SELECTED_THAI_FONT_PATH`: ไฟล์ฟอนต์ไทยที่เลือก
- `_PIL_AVAILABLE`: ว่า Pillow ติดตั้งหรือไม่
- `MODEL_PATH`: ที่อยู่ของ gesture model file
- `_gesture_model`: โมเดล ML ที่โหลด
- `_gesture_label_map`: Mapping จาก class index ไป label
- `UI_LANGUAGE`: ภาษา UI ปัจจุบัน (TH/EN)
- `UI_TEXT`: Dictionary ของข้อความ UI แบบ multilingual
- `THAI_ALPHABET`: List ของตัวอักษรไทย ก-ฮ (32ตัว)
- `_BANMAI_TEMPLATE`: บันทึกท่ามือ "บ้านลับ"

---

## Command Line Arguments (ตัวเลือกสายคำสั่ง)

```
--camera (-c)              : เลือกหมายเลขกล้อง (default: 0)
--video                    : ไฟล์วิดีโอ (optional)
--max-hands                : จำนวนมือสูงสุด (default: 2)
--min-detect-confidence    : ความมั่นใจ (default: 0.5)
--model                    : path ไฟล์ .task model (optional)
--pic-dir                  : โฟลเดอร์ภาพสำหรับการประเมิน
--test                     : ทดสอบ headless (ไม่ต้องกล้อง)
```

---

## ความสัมพันธ์ระหว่างฟังก์ชัน

```
main()
  ├─ init_gesture_model()
  └─ run_on_images() หรือ open_capture()
      ├─ run_with_solutions() / run_with_tasks()
      │   ├─ classify_gesture_model()
      │   │   └─ hand_landmarks_to_vector()
      │   ├─ classify_gesture()
      │   ├─ is_banmai_pose()
      │   │   └─ _normalize_landmarks()
      │   │       └─ _landmark_distance()
      │   └─ _draw_unicode_text()
      └─ run_on_images()
          ├─ _list_image_files()
          ├─ _read_image()
          └─ classify_gesture() / classify_gesture_model()
```

---

## วันที่อัปเดต: 28 มี.ค. 2026
