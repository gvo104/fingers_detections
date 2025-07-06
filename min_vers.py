import cv2
import mediapipe as mp

# Инициализация камеры
cap = cv2.VideoCapture(0)

# Инициализация MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Изменил на 2 для обнаружения обеих рук
    min_tracking_confidence=0.5,
    min_detection_confidence=0.5
)

# Стиль для отрисовки landmarks
mp_drawing = mp.solutions.drawing_utils
hand_landmark_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)

while True:
    ret, img = cap.read()
    if not ret:
        print("Не удалось получить кадр")
        break
    
    # Конвертация BGR в RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Обнаружение рук
    results = hands.process(img_rgb)
    
    # Отрисовка landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Рисуем все landmarks и соединения
            mp_drawing.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                hand_landmark_style)
            
            # Дополнительно отмечаем кончики пальцев
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                # Особо выделяем кончик указательного пальца (id=8)
                if id == 8:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                
                # Отмечаем все точки (меньшего размера)
                cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)  # Здесь была ошибка - добавлена запятая
    
    # Отображение результата
    cv2.imshow("Hand Tracking", img)
    
    # Выход по нажатию 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
cv2.destroyAllWindows()