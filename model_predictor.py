import numpy as np
import pandas as pd
from joblib import load
from collections import deque

class ModelPredictor:
    def __init__(self, model_path):
        self.model = load(model_path)
        self.buffer = deque(maxlen=180)
        self.current_gesture = None

        # Чётко 42 признака — 1 + 5 + (6 * 2 + 6 * 2 + 4 * 2) = 42
        self.feature_names = [
            "all_landmarks_visible",
            "thumb_extended", "index_extended", "middle_extended", "ring_extended", "pinky_extended"
        ]

        for t in ["t0", "t1", "t2"]:
            for part in ["wrist", "index_tip"]:
                self.feature_names.extend([
                    f"{part}_dx_{t}", f"{part}_dy_{t}",
                    f"{part}_vx_{t}", f"{part}_vy_{t}",
                    f"{part}_ax_{t}", f"{part}_ay_{t}"
                ])

    def add_landmarks(self, landmarks):
        if landmarks:
            self.buffer.append(landmarks)

    def predict(self):
        if len(self.buffer) < 180:
            return None

        features = self._extract_features()
        if features:
            df = pd.DataFrame([features], columns=self.feature_names)
            prediction = self.model.predict(df)[0]
            self.current_gesture = self._map_prediction(prediction)
            return self.current_gesture
        return None

    def _extract_features(self):
        if len(self.buffer) < 180:
            return None

        buffer = list(self.buffer)
        frame0 = buffer[0]

        if frame0 is None or any(p is None for p in frame0) or len(frame0) != 21:
            return None

        features = []

        # 1. Статический флаг
        all_points_present = int(all(frame0[i] is not None for i in range(21)))
        features.append(all_points_present)

        # 2. Бинарные признаки: выпрямлены ли пальцы
        def is_finger_extended(lm, tip, pip, mcp, wrist_idx=0):
            wrist = np.array(lm[wrist_idx])
            tip = np.array(lm[tip])
            pip = np.array(lm[pip])
            mcp = np.array(lm[mcp])
            d_tip = np.linalg.norm(tip - wrist)
            d_pip = np.linalg.norm(pip - wrist)
            d_mcp = np.linalg.norm(mcp - wrist)
            return int(d_tip > d_pip and d_tip > d_mcp)

        fingers = {
            'thumb': (4, 3, 2),
            'index': (8, 6, 5),
            'middle': (12, 10, 9),
            'ring': (16, 14, 13),
            'pinky': (20, 18, 17)
        }

        for tip, pip, mcp in fingers.values():
            extended = is_finger_extended(frame0, tip, pip, mcp)
            features.append(extended)

        # 3. Динамика
        wrist_idx = 0
        index_tip_idx = 8
        intervals = [(0, 60), (60, 120), (120, 180)]

        for i, (start, end) in enumerate(intervals):
            A = buffer[start]
            B = buffer[end - 1]
            if A is None or B is None:
                return None

            A = np.array(A)
            B = np.array(B)

            # Смещение (только X и Y)
            wrist_delta = B[wrist_idx][:2] - A[wrist_idx][:2]
            index_delta = B[index_tip_idx][:2] - A[index_tip_idx][:2]
            features.extend([round(x, 5) for x in wrist_delta])
            features.extend([round(x, 5) for x in index_delta])

            # Скорость
            wrist_v = wrist_delta / (end - start)
            index_v = index_delta / (end - start)
            features.extend([round(x, 5) for x in wrist_v])
            features.extend([round(x, 5) for x in index_v])

            # Ускорение
            if i >= 1:
                A_prev = buffer[start - 60]
                B_prev = buffer[end - 61]
                if A_prev is None or B_prev is None:
                    return None

                A_prev = np.array(A_prev)
                B_prev = np.array(B_prev)

                wrist_prev_v = (B_prev[wrist_idx][:2] - A_prev[wrist_idx][:2]) / 60
                index_prev_v = (B_prev[index_tip_idx][:2] - A_prev[index_tip_idx][:2]) / 60

                wrist_acc = (wrist_v - wrist_prev_v) / 60
                index_acc = (index_v - index_prev_v) / 60

                features.extend([round(x, 5) for x in wrist_acc])
                features.extend([round(x, 5) for x in index_acc])
            else:
                features.extend([0.0, 0.0])
                features.extend([0.0, 0.0])

        return features

    def _map_prediction(self, pred):
        gestures = {
            0: None,
            1: "swipe_left",
            2: "swipe_right"
        }
        return gestures.get(pred, None)