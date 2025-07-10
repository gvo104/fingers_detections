import numpy as np

def extract_features_from_window(buffer):
    """Извлекает признаки движения и формы руки (60 кадров)."""
    if len(buffer) < 60:
        return None

    start = buffer[0]
    middle = buffer[30]
    end = buffer[59]

    # === 1. Динамика ===
    delta_wrist = np.array(end[0]) - np.array(start[0])
    delta_x, delta_y = delta_wrist[0], delta_wrist[1]
    delta_total = np.linalg.norm(delta_wrist)
    velocity = delta_total / 60
    angle = np.arctan2(delta_y, delta_x)

    # Δ указательного пальца
    delta_tip2 = np.linalg.norm(np.array(end[8]) - np.array(start[8]))

    dynamic = [delta_x, delta_total, velocity, angle, delta_tip2]

    # === 2. Статика формы руки ===
    wrist = np.array(middle[0])
    fingertips_idx = [4, 8, 12, 16, 20]
    dists = [np.linalg.norm(np.array(middle[i]) - wrist) for i in fingertips_idx]
    d_4, d_8, d_12, d_16, d_20 = dists

    avg_other = np.mean([d_12, d_16, d_20])
    index_ratio = d_8 / (avg_other + 1e-5)  # избежание деления на 0

    static = dists + [index_ratio]

    # Округление до 3 знаков
    all_features = dynamic + static
    return [round(f, 3) for f in all_features]
