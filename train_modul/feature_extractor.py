import numpy as np

def extract_features_from_window(buffer):
    """
    Извлекает признаки из буфера на 180 кадров:
    - Относительные координаты ладони (1-й кадр)
    - Расстояния от запястья до кончиков пальцев
    - Вектор запястье → указательный палец
    - Для трёх сегментов (0-60, 60-120, 120-180):
        - Смещения
        - Средняя скорость (x, y)
        - Среднее ускорение (x, y)
    """

    if len(buffer) < 180:
        return None

    features = []

    # === 1. Относительные координаты landmark'ов (первый кадр) ===
    frame0 = buffer[0]
    if frame0 is None:
        return None

    wrist = np.array(frame0[0])
    rel_coords = []
    for i in range(21):
        point = np.array(frame0[i])
        rel = point - wrist
        rel_coords.extend(rel)

    features.extend([round(x, 5) for x in rel_coords])

    # === 2. Расстояния от запястья до кончиков пальцев (статические признаки) ===
    fingertips_idx = [4, 8, 12, 16, 20]
    dists = [np.linalg.norm(np.array(frame0[i]) - wrist) for i in fingertips_idx]
    features.extend([round(d, 5) for d in dists])

    # === 3. Вектор "запястье → указательный палец" ===
    vec = np.array(frame0[8]) - wrist
    features.extend([round(vec[0], 5), round(vec[1], 5)])

    # === 4. Динамика в трёх временных отрезках ===
    intervals = [(0, 60), (60, 120), (120, 180)]
    for (start, end) in intervals:
        A = buffer[start]
        B = buffer[end - 1]
        if A is None or B is None:
            return None

        A = np.array(A)
        B = np.array(B)

        # Δ (смещение)
        delta = B - A
        features.extend([round(val, 5) for val in delta.flatten()])

        # Скорость (по x и y)
        v = delta / (end - start)
        vx_mean = np.mean(v[:, 0])
        vy_mean = np.mean(v[:, 1])
        features.extend([round(vx_mean, 5), round(vy_mean, 5)])

        # Ускорение (на основе предыдущего интервала)
        if start >= 60:
            A_prev = buffer[start - 60]
            B_prev = buffer[end - 61]
            if A_prev is None or B_prev is None:
                return None
            prev_v = (np.array(B_prev) - np.array(A_prev)) / 60
            acc = (v - prev_v) / 60
            acc_x = np.mean(acc[:, 0])
            acc_y = np.mean(acc[:, 1])
            features.extend([round(acc_x, 5), round(acc_y, 5)])
        else:
            features.extend([0.0, 0.0])

    return features
