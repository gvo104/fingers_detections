import cv2
from hand_tracker import HandTracker
from face_tracker import FaceTracker

def main():
    # Инициализация камеры
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Ошибка: не удалось подключиться к камере!")
        return

    # Получение параметров видео
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Безопасное значение по умолчанию

    # Путь к выходному видеофайлу (замените на свой путь)
    output_path = "F:\\Vs_code_project\\fingers_detections\\results_video\\1.mp4"

    # Инициализация видеозаписи
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Инициализация трекеров
    hand_tracker = HandTracker()
    face_tracker = FaceTracker()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Предупреждение: пропущен кадр")
                continue

            # Зеркальное отражение
            frame = cv2.flip(frame, 1)
            
            # Обработка трекерами
            frame = hand_tracker.process(frame)
            frame = face_tracker.process(frame)
            
            # Отображение подсказки
            cv2.putText(frame, "Q - выход", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Показ результата
            cv2.imshow('Hand & Face Tracking', frame)

            # Запись обработанного кадра в видеофайл
            out.write(frame)
            
            # Выход по клавише Q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Ресурсы освобождены, видео сохранено:", output_path)

if __name__ == "__main__":
    main()
