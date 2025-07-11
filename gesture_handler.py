import time
import pyautogui
import os
from collections import deque

class GestureHandler:
    def __init__(self):
        self.last_open_time = 0
        self.hold_start_time = None
        self.open_triggered = False

        self.gesture_buffer = deque(maxlen=5)
        self.last_swipe_time = 0
        self.swipe_cooldown = 1.0

        self.last_command = "Ожидание жеста..."  # Статус по умолчанию

    def handle_gesture(self, gesture):
        current_time = time.time()

        # Блокировка действий во время "заморозки"
        if current_time - self.last_swipe_time < self.swipe_cooldown:
            return

        if gesture:
            self.gesture_buffer.append(gesture)

        current_gesture = self._get_most_common_gesture()

        # Открытая ладонь
        if current_gesture == "open_palm":
            if self.hold_start_time is None:
                self.hold_start_time = current_time
            elif current_time - self.hold_start_time >= 2 and not self.open_triggered:
                self.open_triggered = True
                self.open_file()
                self.last_command = "Открытие файла"
        else:
            self.hold_start_time = None
            self.open_triggered = False

        # Свайпы
        if current_gesture == "swipe_left":
            self.press_left_arrow()
            self.last_swipe_time = current_time
            self.last_command = "Стрелка влево"
        elif current_gesture == "swipe_right":
            self.press_right_arrow()
            self.last_swipe_time = current_time
            self.last_command = "Стрелка вправо"

    def _get_most_common_gesture(self):
        if not self.gesture_buffer:
            return None

        gesture_counts = {}
        for gesture in self.gesture_buffer:
            gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1

        return max(gesture_counts.items(), key=lambda x: x[1])[0]

    def open_file(self):
        print("Открытие локального файла...")
        file_path = r"D:\1131_GVO_Ispitaniya.pptx"
        try:
            os.startfile(file_path)
        except Exception as e:
            print(f"Ошибка при открытии файла: {e}")

    def press_right_arrow(self):
        print("Свайп вправо -> нажатие стрелки вправо")
        pyautogui.press("right")

    def press_left_arrow(self):
        print("Свайп влево -> нажатие стрелки влево")
        pyautogui.press("left")

    def get_status(self):
        return self.last_command
