import time
import pyautogui
import os
import subprocess

class GestureHandler:
    def __init__(self):
        self.last_open_time = 0
        self.hold_start_time = None
        self.open_triggered = False
        self.right_hold_start = None
        self.left_hold_start = None

    def handle_gesture(self, gesture): 


        current_time = time.time()

        # Логика для open_palm (с флагом, как было)
        if gesture == "open_palm":
            if self.hold_start_time is None:
                self.hold_start_time = current_time
            elif current_time - self.hold_start_time >= 2 and not self.open_triggered:
                self.open_triggered = True
                self.open_file()
        else:
            self.hold_start_time = None
            self.open_triggered = False

        # Логика для point_right (без флага)
        if gesture == "point_right":
            if self.right_hold_start is None:
                self.right_hold_start = current_time
            elif current_time - self.right_hold_start >= 1:
                self.press_right_arrow()
                self.right_hold_start = current_time  # Сбрасываем таймер после действия
        else:
            self.right_hold_start = None

        # Логика для point_left (без флага)
        if gesture == "point_left":
            if self.left_hold_start is None:
                self.left_hold_start = current_time
            elif current_time - self.left_hold_start >= 1:
                self.press_left_arrow()
                self.left_hold_start = current_time  # Сбрасываем таймер после действия
        else:
            self.left_hold_start = None

    def open_file(self):
        print("Открытие локального файла...")
        file_path = r"C:\Users\galki\Pictures\рабочий_стол\andrew-andreev-lake.jpg"
        try:
            os.startfile(file_path)
        except Exception as e:
            print(f"Ошибка при открытии файла: {e}")

    def press_right_arrow(self):
        print("Нажатие стрелки вправо")
        pyautogui.press("right")

    def press_left_arrow(self):
        print("Нажатие стрелки влево")
        pyautogui.press("left")