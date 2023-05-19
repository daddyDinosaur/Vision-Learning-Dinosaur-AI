import pyautogui
import time
import keyboard

num_screenshots = 100
delay = 0.5  

for i in range(num_screenshots):
    screenshot = pyautogui.screenshot(region=(750, 226, 100, 80))
    screenshot.save(f'ai\\validation\class2\\b_{i}.png')
    time.sleep(delay)

# def print_mouse_position(e):
#     print('Mouse position:', pyautogui.position())

# keyboard.on_press_key('a', print_mouse_position)

# keyboard.wait()aa