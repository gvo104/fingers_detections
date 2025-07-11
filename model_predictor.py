import numpy as np
import pandas as pd
from joblib import load
from collections import deque

class ModelPredictor:
    def __init__(self, model_path):
        self.model = load(model_path)
        self.buffer = deque(maxlen=60)
        self.current_gesture = None

        # Имена признаков в том же порядке, в котором они используются в _extract_features()
        self.feature_names = [
            "delta_x", "delta_total", "velocity", "angle", "delta_tip2",
            "dist_thumb", "dist_index", "dist_middle", "dist_ring", "dist_pinky",
            "index_ratio"
        ]

    def add_landmarks(self, landmarks):
        if landmarks:
            self.buffer.append(landmarks)

    def predict(self):
        if len(self.buffer) < 60:
            return None

        features = self._extract_features()
        if features:
            # Оборачиваем данные в DataFrame с указанием имён признаков
            input_df = pd.DataFrame([features], columns=self.feature_names)
            prediction = self.model.predict(input_df)[0]
            self.current_gesture = self._map_prediction(prediction)
            return self.current_gesture
        return None

    def _extract_features(self):
        if len(self.buffer) < 60:
            return None

        # Извлекаем кадры из начала, середины и конца окна
        start = self.buffer[0]
        middle = self.buffer[30]
        end = self.buffer[59]

        # Динамические признаки
        delta_wrist = np.array(end[0]) - np.array(start[0])
        delta_x, delta_y = delta_wrist[0], delta_wrist[1]
        delta_total = np.linalg.norm(delta_wrist)
        velocity = delta_total / 60
        angle = np.arctan2(delta_y, delta_x)
        delta_tip2 = np.linalg.norm(np.array(end[8]) - np.array(start[8]))
        dynamic = [delta_x, delta_total, velocity, angle, delta_tip2]

        # Статические признаки (расстояния от запястья до кончиков пальцев)
        wrist = np.array(middle[0])
        fingertips_idx = [4, 8, 12, 16, 20]
        dists = [np.linalg.norm(np.array(middle[i]) - wrist) for i in fingertips_idx]
        d_4, d_8, d_12, d_16, d_20 = dists

        # Отношение указательного пальца к среднему, безымянному и мизинцу
        avg_other = np.mean([d_12, d_16, d_20])
        index_ratio = d_8 / (avg_other + 1e-5)

        static = dists + [index_ratio]

        all_features = dynamic + static
        return [round(f, 3) for f in all_features]

    def _map_prediction(self, pred):
        gestures = {
            0: None,
            1: "swipe_left",
            2: "swipe_right"
        }
        return gestures.get(pred, None)
