import cv2
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from hand_tracker import HandTracker
from face_tracker import FaceTracker
from gesture_detector import detect_gesture
from gesture_handler import GestureHandler
from model_predictor import ModelPredictor
from put_text_pil import put_text_pil

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Ошибка: не удалось подключиться к камере!")
        return

    hand_tracker = HandTracker()
    face_tracker = FaceTracker()
    gesture_handler = GestureHandler()

    model_path = os.path.join(os.path.dirname(__file__), "train_modul", "CSV", "model.joblib")
    model_predictor = ModelPredictor(model_path)

    face_detection_enabled = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Предупреждение: пропущен кадр")
                continue

            frame = cv2.flip(frame, 1)

            frame, landmarks = hand_tracker.process(frame)

            if face_detection_enabled:
                frame = face_tracker.process(frame)

            gesture = detect_gesture(landmarks, model_predictor)
            gesture_handler.handle_gesture(gesture)

            # Отрисовка текста через Pillow
            frame = put_text_pil(frame, gesture_handler.get_status(), (10, 30), font_size=24, color=(0, 255, 0))
            face_text = "Q - Выход | F - Лицо: " + ("Вкл" if face_detection_enabled else "Выкл")
            frame = put_text_pil(frame, face_text, (10, 60), font_size=24, color=(0, 255, 255))

            cv2.imshow('Hand Gesture Recognition', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                face_detection_enabled = not face_detection_enabled

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Ресурсы освобождены")

if __name__ == "__main__":
    main()
