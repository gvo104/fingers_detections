import cv2
import os
from hand_tracker import HandTracker
from face_tracker import FaceTracker
from gesture_detector import detect_gesture
from gesture_handler import GestureHandler
from model_predictor import ModelPredictor

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Ошибка: не удалось подключиться к камере!")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30

    hand_tracker = HandTracker()
    face_tracker = FaceTracker()
    gesture_handler = GestureHandler()

    model_path = os.path.join(os.path.dirname(__file__), "train_modul", "SCV", "model.joblib")
    model_predictor = ModelPredictor(model_path)

    # Переменная для переключения
    face_detection_enabled = True

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Предупреждение: пропущен кадр")
                continue

            frame = cv2.flip(frame, 1)

            # Рука — всегда
            frame, landmarks = hand_tracker.process(frame)

            # Лицо — по флагу
            if face_detection_enabled:
                frame = face_tracker.process(frame)

            gesture = detect_gesture(landmarks, model_predictor)
            gesture_handler.handle_gesture(gesture)

            # Отображение статуса
            cv2.putText(frame, gesture_handler.get_status(), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Q - Выход | F - Лицо: " + ("Вкл" if face_detection_enabled else "Выкл"), 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow('Hand Gesture Recognition', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                face_detection_enabled = not face_detection_enabled  # Переключение

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Ресурсы освобождены")

if __name__ == "__main__":
    main()
