import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def put_text_pil(img, text, position, font_path='C:\\Windows\\Fonts\\arial.ttf', font_size=24, color=(0, 255, 0)):

    # Конвертация BGR -> RGB для PIL
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)

    # Pillow использует RGB, поэтому меняем порядок цвета
    rgb_color = (color[2], color[1], color[0])

    draw.text(position, text, font=font, fill=rgb_color)

    # Конвертация обратно RGB -> BGR для OpenCV
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
