import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_tracking_confidence=0.5,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.landmark_style = mp.solutions.drawing_utils.DrawingSpec(
            color=(0, 255, 0), thickness=2)
        
    def process(self, img):
        if img is None:
            return img, None  # Теперь возвращается также список координат
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        landmarks_px = None  # список координат в пикселях

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Отрисовка связей
                self.mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.landmark_style)

                # Преобразование нормализованных координат в пиксели
                h, w = img.shape[:2]
                landmarks_px = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks_px.append((cx, cy))

                    # Отрисовка точек
                    if id == 8:  # Кончик указательного пальца
                        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
        
        return img, landmarks_px
