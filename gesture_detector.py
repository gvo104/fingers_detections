import math
from model_predictor import ModelPredictor
import os

# Инициализация модели (один раз)
model_path = os.path.join(os.path.dirname(__file__), "train_modul", "CSV", "model.joblib")
model_predictor = ModelPredictor(model_path)

def detect_gesture(landmarks, predictor=None):
    if landmarks is None:
        return None

    def distance(p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    fingers = []
    tip_ids = [4, 8, 12, 16, 20]
    pip_ids = [3, 6, 10, 14, 18]

    for tip, pip in zip(tip_ids, pip_ids):
        if landmarks[tip][1] < landmarks[pip][1]:
            fingers.append(1)
        else:
            fingers.append(0)

    if fingers == [1, 1, 1, 1, 1]:
        return "open_palm"

    # Используем переданного предсказателя или глобальный
    predictor = predictor or model_predictor
    predictor.add_landmarks(landmarks)
    return predictor.predict()
