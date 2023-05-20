import pyautogui
import time
import keyboard
import pytesseract
import cv2
import easyocr
import numpy

num_screenshots = 100
delay = 0.5  

# for i in range(num_screenshots):
#     screenshot = pyautogui.screenshot(region=(750, 226, 100, 80))
#     screenshot.save(f'ai\\validation\class2\\b_{i}.png')
#     time.sleep(delay)

# def print_mouse_position(e):
#     print('Mouse position:', pyautogui.position())

# keyboard.on_press_key('a', print_mouse_position)

# keyboard.wait()

def get_score():
    screenshot = pyautogui.screenshot(region=(1190, 147, 61, 24))
    screenshot_np = numpy.array(screenshot)

    grayscale_img = cv2.cvtColor(screenshot_np, cv2.COLOR_BGR2GRAY)

    reader = easyocr.Reader(['en'])

    result = reader.readtext(grayscale_img)

    score = int(result[0][-2])
    return score

print(get_score())
