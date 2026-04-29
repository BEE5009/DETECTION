import os
import cv2
import numpy as np
import time
import sys
from PIL import Image, ImageDraw, ImageFont
import pickle
import mediapipe as mp

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


HAND_DETECTOR_TYPE = 'none'

THAI_ALPHABET = [
    'ก', 'ข', 'ค', 'ง', 'จ', 'ฉ', 'ช', 'ซ', 'ฌ', 'ญ',
    'ด', 'ต', 'ถ', 'ท', 'ธ', 'น', 'บ', 'ป', 'ผ', 'พ',
    'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ล', 'ว', 'ศ', 'ษ', 'ส',
    'ห', 'ฮ'
]
ENG_ALPHABET = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
DEFAULT_TASK_MODEL_PATH = 'hand_landmarker.task'
DEFAULT_TASK_MODEL_URL = 'https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task'

HAND_DETECTOR_TYPE = 'none'


def init_hand_detector():
    global HAND_DETECTOR_TYPE

    if hasattr(mp, 'solutions'):
        HAND_DETECTOR_TYPE = 'solutions'
        return

    if hasattr(mp, 'tasks'):
        HAND_DETECTOR_TYPE = 'tasks'
        return

    HAND_DETECTOR_TYPE = 'none'

DATASET_SIZE = 100
COUNTDOWN_SECONDS = 3
VIDEO_START_COUNTDOWN = 4
USE_VIDEO_MODE = True
VIDEO_CLIP_MAX_SECONDS = 30
VIDEO_OUTPUT_FPS = 20
VIDEO_CODEC = 'mp4v'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VIDEO_ROOT_DIR = os.path.join(PROJECT_ROOT, 'data')


data = []
labels = []


brightness_value = 0
contrast_value = 50
is_camera_on = True


def reset_defaults():
    global brightness_value, contrast_value, is_camera_on
    brightness_value = 0
    contrast_value = 50
    is_camera_on = True


def hand_landmarks_to_vector(landmarks):
    if landmarks is None or len(landmarks) != 21:
        return None

    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    min_x = min(xs)
    min_y = min(ys)

    vec = []
    for lm in landmarks:
        vec.append(lm.x - min_x)
        vec.append(lm.y - min_y)

    return np.array(vec, dtype=np.float32).reshape(1, -1)


def safe_read_frame(cap, retries=3):
    if cap is None or not cap.isOpened():
        return False, None

    for attempt in range(retries):
        try:
            ret, frame = cap.read()
        except cv2.error as e:
            print(f" cv2.read failed (attempt {attempt+1}/{retries}): {e}")
            time.sleep(0.05)
            continue

        if not ret or frame is None or frame.size == 0:
            print(f" ได้ frame ไม่ถูกต้อง (attempt {attempt+1}/{retries})")
            time.sleep(0.05)
            continue

        return True, frame

    return False, None


def safe_imshow(window_name, frame, timeout_ms=100):
    try:
        cv2.imshow(window_name, frame)
        return True
    except cv2.error as e:
        print(f" ไม่สามารถแสดงภาพได้: {e}")
        return False


def safe_waitkey(delay=1):
    try:

        key = cv2.waitKey(max(1, delay)) & 0xFF
        return key
    except cv2.error as e:
        print(f" waitKey failed: {e}")
        return -1


def adjust_brightness_contrast(frame, brightness=0, contrast=50):
    if brightness > 0:
        shadow = brightness
        highlight = 255
    else:
        shadow = 0
        highlight = 255 + brightness

    alpha_b = (highlight - shadow) / 255
    gamma_b = shadow

    buf = cv2.convertScaleAbs(frame, alpha=alpha_b, beta=gamma_b)

    contrast_alpha = float(contrast) / 50
    buf = cv2.convertScaleAbs(buf, alpha=contrast_alpha, beta=0)

    return np.clip(buf, 0, 255).astype(np.uint8)


def put_text_unicode(frame, text, pos, font_size=24, color=(0, 255, 0), thickness=1):
    font_path_candidates = [
        "./ฟอนต์/THSarabunPSKv1.0/Fonts TH SarabunPSK v1.0/THSarabunPSK.ttf",
        "./ฟอนต์/THSarabunNew/THSarabunNew.ttf",
        "./ฟอนต์/thai/THSarabunNew/THSarabunNew.ttf"
    ]
    font = None

    for fp in font_path_candidates:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, font_size)
                break
            except Exception:
                font = None

    if font is None:
        font = ImageFont.load_default()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img_pil)
    r, g, b = color
    draw.text(pos, text, font=font, fill=(r, g, b))
    frame_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return frame_bgr


def init_fullscreen_window(window_name):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def draw_hand_landmarks(frame, landmarks):
    if landmarks is None:
        return frame

    h, w = frame.shape[:2]
    points = []
    for lm in landmarks:
        if lm is None:
            continue
        x = int(min(max(lm.x * w, 0), w - 1))
        y = int(min(max(lm.y * h, 0), h - 1))
        points.append((x, y))
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    if len(points) < 2:
        return frame

    try:
        connections = mp.solutions.hands.HAND_CONNECTIONS
    except Exception:
        connections = []

    for conn in connections:
        if conn.start < len(points) and conn.end < len(points):
            start = points[conn.start]
            end = points[conn.end]
            cv2.line(frame, start, end, (0, 255, 255), 2)

    return frame


def create_video_writer(path, frame_size):
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    writer = cv2.VideoWriter(path, fourcc, VIDEO_OUTPUT_FPS, frame_size)
    if not writer.isOpened():
        print(f" ไม่สามารถสร้างไฟล์วิดีโอได้: {path}")
        return None
    return writer


def enhance_image_visibility(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return enhanced


def draw_countdown(frame, countdown_value, text="กำลังจับภาพใน..."):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (w//2 - 200, h//2 - 120), (w//2 + 200, h//2 + 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    frame = put_text_unicode(frame, text, (w//2 - 180, h//2 - 40), font_size=30, color=(0, 255, 0))
    frame = put_text_unicode(frame, str(countdown_value), (w//2 - 40, h//2 + 20), font_size=70, color=(0, 255, 255))
    return frame


def run_video_start_countdown(cap, class_dir, letter, countdown_seconds):
    start_time = time.time()
    while True:
        ret, frame = safe_read_frame(cap)
        if not ret:
            continue

        frame = enhance_image_visibility(frame)
        frame = adjust_brightness_contrast(frame, brightness_value, contrast_value)
        elapsed = time.time() - start_time
        remaining = countdown_seconds - elapsed
        if remaining <= 0:
            return True

        frame = draw_countdown(frame, int(remaining) + 1, f"เริ่มบันทึกวิดีโอ {letter} ใน...")
        frame = draw_controls_info(frame, video_mode=True, recording=False)
        if not safe_imshow('จับภาพตัวอักษร', frame):
            return False

        key = safe_waitkey(100)
        if key == -1:
            continue
        if key == ord('x'):
            return False
        if key in (ord('d'), ord('D')):
            return False
        if key == ord('q'):
            return False

    return False


def draw_collection_status(frame, current_count, total_count=None, elapsed_time=0):
    h, w = frame.shape[:2]
    if total_count is None:
        progress_text = f"คลิปที่บันทึก: {current_count}"
    else:
        progress_text = f"จับภาพ: {current_count}/{total_count}"
    frame = put_text_unicode(frame, progress_text, (10, 40), font_size=28, color=(0,255,0))
    text_time = f"เวลา: {elapsed_time:.1f}s"
    frame = put_text_unicode(frame, text_time, (10, 90), font_size=28, color=(0,255,255))

    if total_count is not None and total_count > 0:
        bar_width = 300
        bar_height = 30
        bar_x = (w - bar_width) // 2
        bar_y = h - 60
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        filled_width = int(bar_width * current_count / total_count)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), (0, 255, 0), -1)
        percent_text = f"{(current_count / total_count * 100):.0f}%"
        frame = put_text_unicode(frame, percent_text, (w//2 - 30, bar_y + 14), font_size=24, color=(255,255,255))
    return frame


def draw_controls_info(frame, video_mode=False, recording=False):
    h, w = frame.shape[:2]
    if video_mode:
        controls = [
            "H=สลับโหมดวิดีโอ/ภาพ | S=เริ่มบันทึก | D=หยุดบันทึก | Q=จบตัวอักษร | X=ออก",
            "R=รีเซ็ต | B/V=สว่าง | C/M=คอนทราสต์ | SPACE=เปิด/ปิดกล้อง"
        ]
    else:
        controls = [
            "H=สลับโหมดวิดีโอ/ภาพ | Q=จับ | 2=ถัดไป | X=ออก | R=รีเซ็ต",
            "B/V=สว่าง | C/M=คอนทราสต์ | SPACE=เปิด/ปิดกล้อง"
        ]
    for i, line in enumerate(controls):
        frame = put_text_unicode(frame, line, (10, h - 60 + i * 30), font_size=20, color=(255,255,0))

    if recording:
        frame = put_text_unicode(frame, "กำลังบันทึกวิดีโอ... (กด D เพื่อหยุด)", (10, h - 110), font_size=28, color=(0,0,255))
    return frame


def wait_for_ready(cap, letter):
    global brightness_value, contrast_value, is_camera_on
    print(f"\n ตัวอักษร {letter} - กำลังรอให้พร้อม...")
    start_time = time.time()
    consecutive_failures = 0
    max_consecutive_failures = 30

    while True:
        if not is_camera_on:
            ret = True
            frame = np.zeros((480, 720, 3), dtype=np.uint8)
            frame = put_text_unicode(frame, "กล้องปิด - กด SPACE เพื่อเปิด", (150, 240), font_size=28, color=(0,0,255))
        else:
            ret, frame = safe_read_frame(cap)
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print(f" กล้องล้มเหลวติดต่อกัน {consecutive_failures} ครั้ง - กด X เพื่อออก หรือ R เพื่อรีเซ็ต")
                cv2.waitKey(100)
                continue
            else:
                consecutive_failures = 0

        if is_camera_on:
            frame = enhance_image_visibility(frame)
            frame = adjust_brightness_contrast(frame, brightness_value, contrast_value)

        h, w = frame.shape[:2]
        text = f'พร้อมจับตัวอักษร "{letter}"? กด Q เพื่อเริ่ม'
        frame = put_text_unicode(frame, text, (30, 50), font_size=26, color=(0,255,0))
        elapsed = time.time() - start_time
        frame = put_text_unicode(frame, f"เวลา: {elapsed:.1f}s", (30, 100), font_size=24, color=(0,255,255))
        frame = draw_controls_info(frame)

        if not safe_imshow('จับภาพตัวอักษร', frame):
            print("❌ ไม่สามารถแสดงหน้าต่างได้ - ออกจากโปรแกรม")
            return 'exit'

        key = safe_waitkey(1)
        if key == -1:
            continue
        if key in (ord('q'), ord('Q')):
            return 'start'
        elif key == ord('2'):
            return 'next'
        elif key == ord('x'):
            cv2.destroyAllWindows(); return 'exit'
        elif key == ord('r'):
            cv2.destroyAllWindows(); reset_defaults(); return 'reset'
        elif key == ord('b'):
            brightness_value = min(100, brightness_value + 5)
        elif key == ord('v'):
            brightness_value = max(0, brightness_value - 5)
        elif key == ord('c'):
            contrast_value = min(100, contrast_value + 5)
        elif key == ord('m'):
            contrast_value = max(0, contrast_value - 5)
        elif key == ord(' '):
            is_camera_on = not is_camera_on


def collect_samples(cap, class_dir, letter, label_idx, hand_detector, hand_detector_type):
    global brightness_value, contrast_value, is_camera_on
    print(f' กำลังจับภาพตัวอักษร {letter}...')
    counter = 0
    total_start_time = time.time()
    consecutive_failures = 0
    max_consecutive_failures = 20

    while counter < DATASET_SIZE:
        if not is_camera_on:
            cv2.waitKey(100)
            continue

        countdown_start = time.time()
        while time.time() - countdown_start < COUNTDOWN_SECONDS:
            ret, frame = safe_read_frame(cap)
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print(f" กล้องล้มเหลวติดต่อกัน {consecutive_failures} ครั้ง - กด X เพื่อออก")
                cv2.waitKey(100)
                continue
            else:
                consecutive_failures = 0

            frame = enhance_image_visibility(frame)
            frame = adjust_brightness_contrast(frame, brightness_value, contrast_value)
            remaining = COUNTDOWN_SECONDS - (time.time() - countdown_start)
            countdown_val = max(1, int(remaining) + 1)
            frame = draw_countdown(frame, countdown_val, f"จับภาพรูปที่ {counter + 1}")
            elapsed_total = time.time() - total_start_time
            frame = draw_collection_status(frame, counter, DATASET_SIZE, elapsed_total)
            frame = draw_controls_info(frame)

            if not safe_imshow('จับภาพตัวอักษร', frame):
                print("❌ ไม่สามารถแสดงหน้าต่างได้ - ออกจากโปรแกรม")
                return 'exit'

            key = safe_waitkey(25)
            if key == -1:
                continue

            if key == ord('q'):
                return 'done'
            elif key == ord('x'):
                cv2.destroyAllWindows(); return 'exit'
            elif key == ord('r'):
                cv2.destroyAllWindows(); reset_defaults(); return 'reset'
            elif key == ord('b'):
                brightness_value = min(100, brightness_value + 5)
            elif key == ord('v'):
                brightness_value = max(0, brightness_value - 5)
            elif key == ord('c'):
                contrast_value = min(100, contrast_value + 5)
            elif key == ord('m'):
                contrast_value = max(0, contrast_value - 5)
            elif key == ord(' '):
                is_camera_on = not is_camera_on

        ret, frame = safe_read_frame(cap)
        if not ret:
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                print(f" กล้องล้มเหลวติดต่อกัน {consecutive_failures} ครั้ง - หยุดการจับภาพ")
                return 'done'
            cv2.waitKey(100)
            continue
        else:
            consecutive_failures = 0

        frame = enhance_image_visibility(frame)
        frame = adjust_brightness_contrast(frame, brightness_value, contrast_value)
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (w//2 - 100, h//2 - 80), (w//2 + 100, h//2 + 80), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        frame = put_text_unicode(frame, "SAVED! ", (w//2 - 90, h//2 + 10), font_size=36, color=(0,255,0))
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)


        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = None

        if hand_detector_type == 'solutions' and hand_detector is not None:
            results = hand_detector.process(image_rgb)
            if results and getattr(results, 'multi_hand_landmarks', None):
                landmarks = results.multi_hand_landmarks[0]

        elif hand_detector_type == 'tasks' and hand_detector is not None:
            try:
                mp_image = mp.Image(mp.ImageFormat.SRGB, image_rgb)
                timestamp_ms = int(time.time() * 1000)
                result = hand_detector.detect_for_video(mp_image, timestamp_ms)
                if result and getattr(result, 'hand_landmarks', None):
                    landmarks = result.hand_landmarks[0]
            except Exception as e:
                print(f" hand detector (tasks) process failed: {e}")
                landmarks = None

        if landmarks is not None:
            vec = hand_landmarks_to_vector(landmarks)
            if vec is not None:
                data.append(vec.flatten())
                labels.append(label_idx)

        counter += 1

        elapsed_total = time.time() - total_start_time
        frame = draw_collection_status(frame, counter, DATASET_SIZE, elapsed_total)
        frame = draw_controls_info(frame)

        if not safe_imshow('จับภาพตัวอักษร', frame):
            print("❌ ไม่สามารถแสดงหน้าต่างได้ - ออกจากโปรแกรม")
            return 'exit'

        key = safe_waitkey(1)
        if key == -1:
            continue

        if key == ord('q'):
            return 'done'
        elif key == ord('2'):
            return 'next'
        elif key == ord('x'):
            cv2.destroyAllWindows(); return 'exit'
        elif key == ord('r'):
            cv2.destroyAllWindows(); reset_defaults(); return 'reset'
        elif key == ord('b'):
            brightness_value = min(100, brightness_value + 5)
        elif key == ord('v'):
            brightness_value = max(0, brightness_value - 5)
        elif key == ord('c'):
            contrast_value = min(100, contrast_value + 5)
        elif key == ord('m'):
            contrast_value = max(0, contrast_value - 5)
        elif key == ord(' '):
            is_camera_on = not is_camera_on

    print(f' เก็บ {counter} รูป ใช้เวลา {time.time() - total_start_time:.1f} วินาที')
    return 'done'


def collect_video_samples(cap, class_dir, letter, label_idx, hand_detector, hand_detector_type):
    global brightness_value, contrast_value, is_camera_on
    print(f' กำลังบันทึกวิดีโอสำหรับตัวอักษร {letter}...')
    clip_counter = 0
    total_start_time = time.time()
    consecutive_failures = 0
    max_consecutive_failures = 20
    recording = False
    video_writer = None
    clip_start_time = 0.0

    init_fullscreen_window('จับภาพตัวอักษร')

    while True:
        if not is_camera_on:
            if recording and video_writer is not None:
                video_writer.release()
                video_writer = None
                recording = False
            cv2.waitKey(100)
            continue

        ret, frame = safe_read_frame(cap)
        if not ret:
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                print(f" กล้องล้มเหลวติดต่อกัน {consecutive_failures} ครั้ง - กด X เพื่อออก")
            cv2.waitKey(100)
            continue

        consecutive_failures = 0
        frame = enhance_image_visibility(frame)
        frame = adjust_brightness_contrast(frame, brightness_value, contrast_value)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        landmarks = None
        if hand_detector_type == 'solutions' and hand_detector is not None:
            results = hand_detector.process(image_rgb)
            if results and getattr(results, 'multi_hand_landmarks', None):
                landmarks = results.multi_hand_landmarks[0]
        elif hand_detector_type == 'tasks' and hand_detector is not None:
            try:
                mp_image = mp.Image(mp.ImageFormat.SRGB, image_rgb)
                timestamp_ms = int(time.time() * 1000)
                result = hand_detector.detect_for_video(mp_image, timestamp_ms)
                if result and getattr(result, 'hand_landmarks', None):
                    landmarks = result.hand_landmarks[0]
            except Exception as e:
                print(f" hand detector (tasks) process failed: {e}")
                landmarks = None

        if landmarks is not None:
            frame = draw_hand_landmarks(frame, landmarks)
            if recording:
                vec = hand_landmarks_to_vector(landmarks)
                if vec is not None:
                    data.append(vec.flatten())
                    labels.append(label_idx)

        if recording and video_writer is not None:
            video_writer.write(frame)
            elapsed_clip = time.time() - clip_start_time
            frame = put_text_unicode(frame, f"บันทึกคลิป {clip_counter + 1} เวลา {elapsed_clip:.1f}s", (10, 120), font_size=26, color=(0,255,0))
            if elapsed_clip >= VIDEO_CLIP_MAX_SECONDS:
                video_writer.release()
                video_writer = None
                recording = False
                clip_counter += 1
                print(f" หยุดบันทึกอัตโนมัติ หลัง {VIDEO_CLIP_MAX_SECONDS} วินาที")
        else:
            frame = put_text_unicode(frame, "กด S เพื่อเริ่มบันทึกวิดีโอ | D=หยุด | Q=จบตัวอักษร", (10, 120), font_size=26, color=(255,255,0))

        frame = put_text_unicode(frame, f"ตัวอักษร: {letter}", (10, 70), font_size=28, color=(0,255,255))
        elapsed_total = time.time() - total_start_time
        frame = draw_collection_status(frame, clip_counter, None, elapsed_total)
        frame = draw_controls_info(frame, video_mode=True, recording=recording)

        if not safe_imshow('จับภาพตัวอักษร', frame):
            print("❌ ไม่สามารถแสดงหน้าต่างได้ - ออกจากโปรแกรม")
            if recording and video_writer is not None:
                video_writer.release()
            return 'exit'

        key = safe_waitkey(1)
        if key == -1:
            continue

        if key in (ord('s'), ord('S')) and not recording:
            started = run_video_start_countdown(cap, class_dir, letter, VIDEO_START_COUNTDOWN)
            if not started:
                continue
            video_name = f"video_{letter}_{int(time.time())}_{clip_counter + 1}.mp4"
            rel_dir = os.path.relpath(class_dir, './data').replace('\\', '/')
            if rel_dir.startswith('thai'):
                class_subdir = os.path.relpath(class_dir, './data/thai')
                video_dir = os.path.join(VIDEO_ROOT_DIR, 'thai_video', class_subdir)
            elif rel_dir.startswith('eng'):
                class_subdir = os.path.relpath(class_dir, './data/eng')
                video_dir = os.path.join(VIDEO_ROOT_DIR, 'eng_video', class_subdir)
            else:
                video_dir = os.path.join(VIDEO_ROOT_DIR, 'videos', rel_dir)
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, video_name)
            video_writer = create_video_writer(video_path, (frame.shape[1], frame.shape[0]))
            if video_writer is not None:
                recording = True
                clip_start_time = time.time()
                print(f" เริ่มบันทึกวิดีโอ: {video_name}")
            continue

        if key in (ord('d'), ord('D')) and recording:
            if video_writer is not None:
                video_writer.release()
                video_writer = None
            recording = False
            clip_counter += 1
            print(f" หยุดบันทึกคลิป {clip_counter}")
            continue

        if key == ord('q'):
            if recording and video_writer is not None:
                video_writer.release()
                video_writer = None
            return 'done'
        elif key == ord('2'):
            if recording and video_writer is not None:
                video_writer.release()
                video_writer = None
            return 'next'
        elif key == ord('x'):
            if recording and video_writer is not None:
                video_writer.release()
                video_writer = None
            cv2.destroyAllWindows(); return 'exit'
        elif key == ord('r'):
            if recording and video_writer is not None:
                video_writer.release()
                video_writer = None
            recording = False
            reset_defaults()
            return 'reset'
        elif key == ord('b'):
            brightness_value = min(100, brightness_value + 5)
        elif key == ord('v'):
            brightness_value = max(0, brightness_value - 5)
        elif key == ord('c'):
            contrast_value = min(100, contrast_value + 5)
        elif key == ord('m'):
            contrast_value = max(0, contrast_value - 5)
        elif key == ord(' '):
            is_camera_on = not is_camera_on

    return 'done'


def main():
    global is_camera_on, HAND_DETECTOR_TYPE, USE_VIDEO_MODE

    init_hand_detector()

    print("\n" + "="*70)
    print(" ระบบรวบรวมข้อมูลสำหรับตรวจจับภาษามือ")
    print("="*70)
    print("โปรดตรวจสอบก่อนเริ่ม:")
    print(" กล้อง webcam เชื่อมต่อและไม่ถูกใช้งานโดยโปรแกรมอื่น")
    print(" ติดตั้ง dependencies จาก requirements.txt")
    print(" มีพื้นที่ว่างในดิสก์เพียงพอ")
    print("="*70 + "\n")

    while True:
        print(f"จำนวนรูปต่อตัวอักษร: {DATASET_SIZE} รูป")
        mode_name = 'วิดีโอ' if USE_VIDEO_MODE else 'ภาพนิ่ง'
        print(f"โหมดการเก็บข้อมูล: {mode_name}")
        print(f"เวลาอยู่ระหว่างจับ: {COUNTDOWN_SECONDS} วินาที")
        print("\n ปุ่มควบคุม:")
        print("   H = สลับโหมดวิดีโอ/ภาพนิ่ง")
        print("   Q = เริ่ม/ข้าม")
        print("   S = เริ่มบันทึก (ในโหมดวิดีโอ)")
        print("   D = หยุดบันทึก (ในโหมดวิดีโอ)")
        print("   B/V = เพิ่ม/ลดความสว่าง")
        print("   C/M = เพิ่ม/ลด Contrast")
        print("   X = ปิดจอ")
        print("   R = รีเซ็ตทั้งหมด")
        print("   SPACE = เปิด/ปิดกล้อง")
        print("="*60 + "\n")

        cap = None
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
        if hasattr(cv2, 'CAP_V4L'):
            backends.append(cv2.CAP_V4L)
        consecutive_failures = 0
        max_consecutive_failures = 50

        for idx in range(5):
            for backend in backends:
                cap_try = cv2.VideoCapture(idx, backend)
                if not cap_try.isOpened():
                    cap_try.release()
                    continue
                ret, frame = safe_read_frame(cap_try)
                if ret:
                    cap = cap_try
                    print(f" เปิดกล้องสำเร็จ (Camera {idx}, backend={backend})\n")
                    break
                cap_try.release()
            if cap is not None:
                break

        if cap is None:
            print(" ไม่สามารถเปิดกล้องได้ - ตรวจสอบ:")
            print("  - กล้องเชื่อมต่อและไม่ถูกใช้งานโดยโปรแกรมอื่น")
            print("  - ติดตั้งไดรเวอร์กล้อง")
            print("  - ลองปิดและเปิดโปรแกรมนี้ใหม่")
            print("  - ตรวจสอบว่า webcam ไม่ถูกปิดใช้งานใน Device Manager")
            print("\nกด Enter เพื่อลองใหม่ หรือพิมพ์ 'exit' เพื่อออก...")
            try:
                user_input = input().strip().lower()
                if user_input == 'exit':
                    return
                else:
                    continue
            except KeyboardInterrupt:
                return

        is_camera_on = True
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        hand_detector = None
        if HAND_DETECTOR_TYPE == 'solutions':
            try:
                hand_detector = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
            except Exception as e:
                print(f" mediapipe solutions init failed: {e}")
                HAND_DETECTOR_TYPE = 'none'
        elif HAND_DETECTOR_TYPE == 'tasks':
            try:
                from mediapipe.tasks.python.vision import hand_landmarker as hl_module
                from mediapipe.tasks.python.core.base_options import BaseOptions
                from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
                from mediapipe.tasks.python.vision.core import vision_task_running_mode as vrm

                model_path = DEFAULT_TASK_MODEL_PATH if os.path.exists(DEFAULT_TASK_MODEL_PATH) else DEFAULT_TASK_MODEL_URL
                base_options = BaseOptions(model_asset_path=model_path)
                running_mode = getattr(vrm.VisionTaskRunningMode, 'VIDEO', vrm.VisionTaskRunningMode.IMAGE)
                options = HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=running_mode,
                    num_hands=1,
                    min_hand_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                hand_detector = HandLandmarker.create_from_options(options)
            except Exception as e:
                print(f" mediapipe tasks init failed: {e}")
                HAND_DETECTOR_TYPE = 'none'

        if HAND_DETECTOR_TYPE == 'none':
            print(' mediapipe ไม่พบโหมดที่รองรับ (solutions/tasks). การบันทึก landmarks จะไม่ทำงาน')

        alphabet = THAI_ALPHABET
        lang_name = "ไทย"
        data_dir = './data/thai'

        while True:
            ret, frame = safe_read_frame(cap)
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print(f" กล้องล้มเหลวติดต่อกัน {consecutive_failures} ครั้ง - อาจมีปัญหากับการเชื่อมต่อกล้อง")
                    print("กด R เพื่อรีเซ็ต หรือ X เพื่อออก")
                cv2.waitKey(100)
                continue
            else:
                consecutive_failures = 0

            h, w = frame.shape[:2]
            frame = put_text_unicode(frame, f"ภาษา: {lang_name}", (w//2 - 100, h//2 - 50), font_size=32, color=(0,255,0))
            frame = put_text_unicode(frame, "กด A เพื่อเปลี่ยน | Q เพื่อเริ่ม | X=ปิด", (w//2 - 220, h//2 + 50), font_size=24, color=(255,255,0))

            if not safe_imshow('เลือกภาษา', frame):
                print(" ไม่สามารถแสดงหน้าต่างได้ - ออกจากโปรแกรม")
                cap.release()
                cv2.destroyAllWindows()
                return

            key = safe_waitkey(0)
            if key == -1:
                continue

            if key in (ord('a'), ord('A')):
                if alphabet == THAI_ALPHABET:
                    alphabet = ENG_ALPHABET
                    lang_name = "อังกฤษ"
                    data_dir = './data/eng'
                else:
                    alphabet = THAI_ALPHABET
                    lang_name = "ไทย"
                    data_dir = './data/thai'
                print(f" เปลี่ยนเป็นภาษา: {lang_name}")
            elif key in (ord('q'), ord('Q')):
                break
            elif key == ord('x'):
                print("\n ปิดโปรแกรม")
                cap.release(); cv2.destroyAllWindows(); return

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)


        for idx in range(len(alphabet)):
            os.makedirs(os.path.join(data_dir, str(idx)), exist_ok=True)

        overall_start_time = time.time()
        current_idx = 0

        while True:
            letter = alphabet[current_idx]
            ret, frame = safe_read_frame(cap)
            if not ret:
                continue

            h, w = frame.shape[:2]
            frame = put_text_unicode(frame, f"ภาษา: {lang_name}", (10, 30), font_size=24, color=(0,255,0))
            mode_text = 'วิดีโอ' if USE_VIDEO_MODE else 'ภาพนิ่ง'
            frame = put_text_unicode(frame, f"โหมด: {mode_text}", (10, 190), font_size=24, color=(0,255,0))
            frame = put_text_unicode(frame, f"ตัวอักษรปัจจุบัน: {letter} ({current_idx+1}/{len(alphabet)})", (10, 70), font_size=28, color=(0,255,255))
            frame = put_text_unicode(frame, "H=สลับโหมด | Q=จับ | 1=ถัดไป | 0=ก่อนหน้า | 2=ข้าม+จับ | X=ออก", (10, 110), font_size=22, color=(255,255,0))
            frame = put_text_unicode(frame, "S/D ใช้เฉพาะโหมดวิดีโอ | R=รีเซ็ต | B/V=สว่าง | C/M=คอนทราสต์", (10, 150), font_size=20, color=(255,255,0))
            safe_imshow('เลือกตัวอักษร', frame)

            key = safe_waitkey(0)
            if key == -1:
                continue

            if key in (ord('a'), ord('A')):
                if alphabet == THAI_ALPHABET:
                    alphabet = ENG_ALPHABET
                    lang_name = 'อังกฤษ'
                    data_dir = './data/eng'
                else:
                    alphabet = THAI_ALPHABET
                    lang_name = 'ไทย'
                    data_dir = './data/thai'
                current_idx = 0
                continue

            if key in (ord('h'), ord('H')):
                USE_VIDEO_MODE = not USE_VIDEO_MODE
                mode_name = 'วิดีโอ' if USE_VIDEO_MODE else 'ภาพนิ่ง'
                print(f" เปลี่ยนโหมดเป็น: {mode_name}")
                continue

            if key == ord('x'):
                print("\n ปิดโปรแกรม")
                break

            if key == ord('v'):
                brightness_value = max(0, brightness_value - 5)
                continue

            if key == ord('r'):
                reset_defaults()
                continue

            if key == ord('b'):
                brightness_value = min(100, brightness_value + 5)
                continue
            if key == ord('v'):
                brightness_value = max(0, brightness_value - 5)
                continue
            if key == ord('c'):
                contrast_value = min(100, contrast_value + 5)
                continue
            if key == ord('m'):
                contrast_value = max(0, contrast_value - 5)
                continue

            if key == ord('1'):
                current_idx = (current_idx + 1) % len(alphabet)
                continue
            if key == ord('0'):
                current_idx = (current_idx - 1) % len(alphabet)
                continue

            if key == ord('v'):
                brightness_value = max(0, brightness_value - 5)
                continue

            if key == ord('q') or key == ord('2'):
                if key == ord('2'):
                    current_idx = (current_idx + 1) % len(alphabet)
                    letter = alphabet[current_idx]

                class_dir = os.path.join(data_dir, str(current_idx))
                try:
                    if USE_VIDEO_MODE:
                        action = collect_video_samples(cap, class_dir, letter, current_idx, hand_detector, HAND_DETECTOR_TYPE)
                    else:
                        action = collect_samples(cap, class_dir, letter, current_idx, hand_detector, HAND_DETECTOR_TYPE)
                except Exception as e:
                    print(f" เกิดข้อผิดพลาดใน collect_samples: {e}")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                if action == 'exit':
                    cap.release(); cv2.destroyAllWindows(); return
                if action == 'reset':
                    reset_defaults()
                    continue

                current_idx = (current_idx + 1) % len(alphabet)
                continue

        with open('data.pickle', 'wb') as f:
            pickle.dump({'data': np.array(data), 'labels': np.array(labels)}, f)
        print('บันทึก data.pickle สำเร็จสำหรับการฝึกโมเดล')

        overall_elapsed = time.time() - overall_start_time
        cap.release()
        cv2.destroyAllWindows()

        print('\n' + '='*60)
        print(f' เสร็จสิ้น! ใช้เวลารวม: {overall_elapsed:.1f} วินาที ({overall_elapsed/60:.1f} นาที)')
        print(f'บันทึกข้อมูลไว้ใน: {data_dir}')
        print(f'ภาษา: {lang_name}')
        if hand_detector is not None:
            try:
                hand_detector.close()
            except Exception:
                pass
        print('='*60 + '\n')
        break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n ผู้ใช้กด Ctrl+C - ออกจากโปรแกรม")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n\n เกิดข้อผิดพลาดที่ไม่คาดคิด: {e}")
        print("โปรดตรวจสอบ:")
        print("- กล้องเชื่อมต่อและใช้งานได้")
        print("- ติดตั้ง dependencies ครบถ้วน")
        print("- ไม่มีโปรแกรมอื่นใช้กล้อง")
        cv2.destroyAllWindows()
    finally:
        cv2.destroyAllWindows()
