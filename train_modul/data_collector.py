import cv2
import numpy as np
from collections import deque
from hand_tracker import HandTracker
from feature_extractor import extract_features_from_window
from data_saver import DataSaver
from PIL import Image, ImageDraw, ImageFont
import os
import time

# Путь к CSV-файлу (будет перезаписываться каждый запуск)
CSV_FILE = "train_modul/CSV/collected_features.csv"
BUFFER_SIZE = 180  # окно в 180 кадров для анализа

# Управление метками
LABEL_KEYS = {
    ord('0'): 0,  # none
    ord('1'): 1,  # swipe_left
    ord('2'): 2   # swipe_right
}
LABEL_NAMES = {
    0: 'none',
    1: 'swipe_left',
    2: 'swipe_right'
}

current_label = None
last_save_time = None

# ===== Отрисовка текста с помощью PIL =====
def put_text_pil(img, text, position, font_path='C:\\Windows\\Fonts\\arial.ttf', font_size=24, color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    # Цвет RGB для PIL
    rgb_color = (color[2], color[1], color[0])
    draw.text(position, text, font=font, fill=rgb_color)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ==========================================

def main():
    global current_label, last_save_time

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Ошибка: камера не найдена.")
        return

    tracker = HandTracker()
    buffer = deque(maxlen=BUFFER_SIZE)

    saver = DataSaver(CSV_FILE, batch_size=10)

    print("Нажмите 0 — none, 1 — swipe_left, 2 — swipe_right. Q — выход.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            frame, landmarks = tracker.process(frame)

            # === Вывод метки ===
            label_text = f"Label: {LABEL_NAMES.get(current_label, '-')}"
            frame = put_text_pil(frame, label_text, (10, 20), color=(0, 0, 255))

            # === Вывод уведомления о сохранении ===
            if last_save_time and time.time() - last_save_time < 1.5:
                frame = put_text_pil(frame, "Сохранено!", (10, 50), color=(0, 200, 0))

            if landmarks is not None:
                buffer.append(landmarks)

            # === Запись данных ===
            if len(buffer) == BUFFER_SIZE and current_label is not None:
                if all(frame is not None for frame in buffer):
                    features = extract_features_from_window(buffer)
                    if features:
                        row = [current_label] + features
                        saver.add_row(row)
                        print(f"[INFO] Записан фрагмент с меткой {LABEL_NAMES[current_label]}")
                        last_save_time = time.time()
                    else:
                        print("[WARN] Признаки не извлечены.")
                else:
                    print("[WARN] Буфер содержит пустые кадры.")
                current_label = None  # сброс

            cv2.imshow("Gesture Data Collector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in LABEL_KEYS:
                current_label = LABEL_KEYS[key]
                print(f"[INFO] Установлена метка: {LABEL_NAMES[current_label]}")
            elif key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        saver.close()
        print("Сбор завершён. Данные сохранены в", CSV_FILE)

if __name__ == "__main__":
    main()
