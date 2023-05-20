import mss
import pyautogui
import cv2
import numpy
import time
import multiprocessing
import tensorflow as tf
from keras.preprocessing.image import image_utils
from PIL import Image
import settings
import easyocr

class ScreenCaptureAgent:
    def __init__(self) -> None:  
        self.capture_process = None
        self.fps = None
        self.enable_preview = True

        self.w, self.h = pyautogui.size()
        print("Screen size: ", self.w, self.h)
        self.monitor = {
            "top": settings.COMP_VIZ_TOP_LEFT[1], 
            "left": settings.COMP_VIZ_TOP_LEFT[0], 
            "width": settings.COMP_VIZ_BOTTOM_RIGHT[0], 
            "height": settings.COMP_VIZ_BOTTOM_RIGHT[1]
        }
        self.model = tf.keras.models.load_model('model_a.h5')
        self.start_time = time.time()

    def capture_screen(self):
        with mss.mss() as sct:
            while True:
                self.img = numpy.array(sct.grab(self.monitor))

                img = Image.fromarray(self.img)
                img = img.resize((80, 100))
                
                img = img.convert('RGB')

                img_array = numpy.array(img) / 255.0
                img_array = numpy.expand_dims(img_array, axis=0)  

                result = self.model.predict(img_array)
                if result[0][0] < 0.5:
                    #make it get its own training screenshots xd
                    screenshot = pyautogui.screenshot(region=(750, 226, 100, 80))
                    screenshot.save(f'ai\\validation\storage\\a_{result[0][0]}.png')

                    elapsed_time = time.time() - self.start_time  
                    sleep_time = max(0.03 - elapsed_time / 1000, 0.005)  
                    time.sleep(sleep_time)
                    pyautogui.press('up')
                else:
                    action = 'NOT JUMP'

                # if self.enable_preview:
                #     preview = cv2.resize(self.img, (0, 0), fx=0.5, fy=0.5)
                #     cv2.imshow("preview", preview)
                #     if cv2.waitKey(25) & 0xFF == ord("q"):
                #         cv2.destroyAllWindows()
                #         break

if __name__ == "__main__":
    agent = ScreenCaptureAgent()
    agent.capture_screen()
