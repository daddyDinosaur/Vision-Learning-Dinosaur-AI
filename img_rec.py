import numpy as np
from PIL import Image
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os

img = Image.open('ai\\validation\class1\\a_1.4378993000718765e-05.png')
img = img.resize((80, 100))

img = np.array(img) / 255.0

img = np.expand_dims(img, axis=0)

new_model = tf.keras.models.load_model('model_a.h5')
result = new_model.predict(img)
print(result)
if result[0][0] < 0.5:
    prediction = 'class_1'
else:
    prediction = 'class_2'

print(prediction)
