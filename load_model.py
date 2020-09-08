import os
import tempfile
import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)
"""
img_path = './3439898752,234818676.jpg'
img_i = tf.keras.preprocessing.image.load_img(img_path)
img_i = np.asarray(img_i)
#img_tensor = tf.cast(img_i, tf.uint8) 
#img_tensor = tf.cast(img_i, tf.float32) 
#print(img_tensor.shape)
"""
model_save_path='SavedModel/efn_b7_epoch_15_acc82_Noshuffle/'
loaded = tf.saved_model.load(model_save_path)
print(list(loaded.signatures.keys()))  # ["serving_default"]
infer = loaded.signatures["serving_default"]
print(infer.structured_outputs)

flag_img = 1
if flag_img:
    
    img_path = './3439898752,234818676.jpg'
    img_i = tf.keras.preprocessing.image.load_img(img_path)
    img_i = np.asarray(img_i)
    img_tensor = tf.cast(img_i, tf.float32) 
else:

    img = np.ones([600, 600, 3])
    b = np.ones([600, 600])
    g = b * 2
    r = b * 3
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    img_tensor = tf.cast(img, tf.float32) 

print("---------------"*3)
value = infer(img_tensor)
print("[*] value: ")
print(value)
print("[*] value numpy: ")
value_numpy = np.asarray(value["OutputLayer"])
print(value_numpy)
print("[*] test load model is done!")

