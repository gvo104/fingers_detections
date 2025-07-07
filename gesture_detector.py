import math

def detect_gesture(landmarks):
    if landmarks is None:
        return None

    def distance(p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    fingers = []

    # Идентификаторы концов пальцев и промежуточных суставов
    tip_ids = [4, 8, 12, 16, 20]   # Большой, указательный, средний, безымянный, мизинец
    pip_ids = [3, 6, 10, 14, 18]

    for tip, pip in zip(tip_ids, pip_ids):
        if landmarks[tip][1] < landmarks[pip][1]:  # Если кончик пальца выше сустава — палец поднят
            fingers.append(1)
        else:
            fingers.append(0)

    # === Определения жестов ===

    # 1. Открытая ладонь: все пальцы подняты
    if fingers == [1, 1, 1, 1, 1]:
        return "open_palm"

    # 2. Указательный и большой подняты, остальные сжаты
    if fingers == [1, 1, 0, 0, 0]:
        return "point_right"

    # 3. Большой и мизинец подняты
    if fingers == [1, 1, 1, 0, 0]:
        return "point_left"

    # Можно добавлять и другие жесты
    return None
