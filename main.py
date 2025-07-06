import cv2
from hand_tracker import HandTracker
from face_tracker import FaceTracker
from gesture_detector import detect_gesture
from gesture_handler import GestureHandler

def main():
    # Инициализация камеры
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Ошибка: не удалось подключиться к камере!")
        return

    # Получение параметров видео
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Безопасное значение по умолчанию

    # Путь к выходному видеофайлу (замените на свой путь)
    output_path = "F:\\Vs_code_project\\fingers_detections\\results_video\\1.mp4"

    # Инициализация видеозаписи
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Инициализация трекеров
    hand_tracker = HandTracker()
    face_tracker = FaceTracker()
    gesture_handler = GestureHandler()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Предупреждение: пропущен кадр")
                continue

            # Зеркальное отражение
            frame = cv2.flip(frame, 1)

            # Обработка руки и получение координат
            frame, landmarks = hand_tracker.process(frame)

            # Обработка лица (если нужно)
            frame = face_tracker.process(frame)

            # Определение жеста и выполнение действия
            gesture = detect_gesture(landmarks)
            gesture_handler.handle_gesture(gesture)

            # Отображение подсказки
            cv2.putText(frame, "Q - выход", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Показ результата
            cv2.imshow('Hand & Face Tracking', frame)

            # Запись кадра
            out.write(frame)

            # Выход по клавише Q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Ресурсы освобождены, видео сохранено:", output_path)

if __name__ == "__main__":
    main()
