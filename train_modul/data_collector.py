import cv2
from collections import deque
from hand_tracker import HandTracker
from feature_extractor import extract_features_from_window
from data_saver import DataSaver
import os

# Путь к CSV-файлу (будет перезаписываться каждый запуск)
CSV_FILE = "train_module/CSV/collected_features.csv"
BUFFER_SIZE = 60  # окно из 60 кадров

# Управление метками
LABEL_KEYS = {
    ord('0'): 0,  # none
    ord('1'): 1,  # swipe_left
    ord('2'): 2   # swipe_right
}
current_label = None  # по умолчанию — ничего не выбрано

def main():
    global current_label
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Ошибка: камера не найдена.")
        return

    tracker = HandTracker()
    buffer = deque(maxlen=BUFFER_SIZE)

    # Очистим и создадим CSV-файл заново
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
    with open(CSV_FILE, 'w') as f:
        f.write("label," + ",".join([f"f{i}" for i in range(1, 26 + 1)]) + "\n")  # 26 признаков

    saver = DataSaver(CSV_FILE, batch_size=10)

    print("Нажмите 0 - none, 1 - swipe_left, 2 - swipe_right. Q — выход.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            frame, landmarks = tracker.process(frame)

            # Показываем текущую метку на экране
            label_text = f"Label: {current_label if current_label is not None else '-'}"
            cv2.putText(frame, label_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if landmarks is not None:
                buffer.append(landmarks)

            # Если метка выбрана и буфер заполнен — извлекаем признаки и сохраняем
            if current_label is not None and len(buffer) == BUFFER_SIZE:
                features = extract_features_from_window(buffer)
                if features:
                    row = [current_label] + features
                    saver.add_row(row)

            cv2.imshow("Gesture Data Collector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in LABEL_KEYS:
                current_label = LABEL_KEYS[key]
                print(f"[INFO] Текущая метка: {current_label}")
            elif key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        saver.close()
        print("Сбор завершён. Данные сохранены в", CSV_FILE)

if __name__ == "__main__":
    main()
