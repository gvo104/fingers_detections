import numpy as np

def is_finger_extended(lm, tip_idx, pip_idx, mcp_idx, wrist_idx=0):
    """
    Простейшая эвристика: палец считается выпрямленным,
    если расстояние (wrist → tip) больше, чем (wrist → pip) и (wrist → mcp)
    """
    wrist = np.array(lm[wrist_idx])
    tip = np.array(lm[tip_idx])
    pip = np.array(lm[pip_idx])
    mcp = np.array(lm[mcp_idx])

    d_tip = np.linalg.norm(tip - wrist)
    d_pip = np.linalg.norm(pip - wrist)
    d_mcp = np.linalg.norm(mcp - wrist)

    return int(d_tip > d_pip and d_tip > d_mcp)

def extract_features_from_window(buffer):
    if len(buffer) < 180:
        return None

    features = []

    # === Проверка, что landmarks в первом кадре есть и корректны ===
    frame0 = buffer[0]
    if frame0 is None or any(p is None for p in frame0) or len(frame0) != 21:
        return None

    # === 1. Статические признаки ===

    # 1.1 Флаг: все ли landmarks присутствуют
    all_points_present = int(all(frame0[i] is not None for i in range(21)))
    features.append(all_points_present)

    # 1.2 Выпрямлены ли пальцы (бинарно)
    # Индексы по структуре MediaPipe
    fingers = {
        'thumb':  (4, 3, 2),
        'index':  (8, 6, 5),
        'middle': (12, 10, 9),
        'ring':   (16, 14, 13),
        'pinky':  (20, 18, 17)
    }
    for tip, pip, mcp in fingers.values():
        extended = is_finger_extended(frame0, tip, pip, mcp)
        features.append(extended)

    # === 2. Динамика кисти и указательного пальца ===

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

        # Смещение
        wrist_delta = B[wrist_idx] - A[wrist_idx]
        index_delta = B[index_tip_idx] - A[index_tip_idx]

        features.extend([round(x, 5) for x in wrist_delta])
        features.extend([round(x, 5) for x in index_delta])

        # Скорость
        wrist_v = wrist_delta / (end - start)
        index_v = index_delta / (end - start)
        features.extend([round(x, 5) for x in wrist_v])
        features.extend([round(x, 5) for x in index_v])

        # Ускорение — только если есть предыдущий интервал
        if i >= 1:
            A_prev = buffer[start - 60]
            B_prev = buffer[end - 61]

            if A_prev is None or B_prev is None:
                return None

            A_prev = np.array(A_prev)
            B_prev = np.array(B_prev)

            wrist_prev_v = (B_prev[wrist_idx] - A_prev[wrist_idx]) / 60
            index_prev_v = (B_prev[index_tip_idx] - A_prev[index_tip_idx]) / 60

            wrist_acc = (wrist_v - wrist_prev_v) / 60
            index_acc = (index_v - index_prev_v) / 60

            features.extend([round(x, 5) for x in wrist_acc])
            features.extend([round(x, 5) for x in index_acc])
        else:
            # Заполняем нулями ускорение для первого интервала
            features.extend([0.0, 0.0, 0.0, 0.0])

    return features
