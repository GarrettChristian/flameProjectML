import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np


from PIL import Image
from PIL import ImageOps
from PIL import ImageEnhance

from sklearn.model_selection import train_test_split

from tensorflow import keras
import random

from keras.models import Sequential

import glob

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Adapted from:
# https://www.tensorflow.org/tutorials/images/cnn
# https://colab.research.google.com/drive/1nseete5huZlWM7Ak0qL-T75Dbk0mdr-Z?usp=sharing#scrollTo=o-qUPyfO7Qr8
# https://github.com/AlirezaShamsoshoara/Fire-Detection-UAV-Aerial-Image-Classification-Segmentation-UnmannedAerialVehicle/blob/30ae2526a5e0e744701f24bb44a2de62e355aece/training.py


bnmomemtum=0.9
def fire(x, squeeze, expand):
    y  = tf.keras.layers.Conv2D(filters=squeeze, kernel_size=1, activation='relu', padding='same')(x)
    y = tf.keras.layers.BatchNormalization(momentum=bnmomemtum)(y)
    y1 = tf.keras.layers.Conv2D(filters=expand//2, kernel_size=1, activation='relu', padding='same')(y)
    y1 = tf.keras.layers.BatchNormalization(momentum=bnmomemtum)(y1)
    y3 = tf.keras.layers.Conv2D(filters=expand//2, kernel_size=3, activation='relu', padding='same')(y)
    y3 = tf.keras.layers.BatchNormalization(momentum=bnmomemtum)(y3)
    return tf.keras.layers.concatenate([y1, y3])

def fire_module(squeeze, expand):
    return lambda x: fire(x, squeeze, expand)
    
def getModel():

    # Adapted from
    # https://colab.research.google.com/github/GoogleCloudPlatform/training-data-analyst/blob/master/courses/fast-and-lean-data-science/07_Keras_Flowers_TPU_squeezenet.ipynb#scrollTo=XLJNVGwHUDy1
    # Squeeze net
    # https://github.com/rcmalli/keras-squeezenet/blob/master/keras_squeezenet/squeezenet.py

    inputs = layers.Input(shape=(254, 254, 3))

    y = inputs

    y = tf.keras.layers.Conv2D(kernel_size=3, filters=32, padding='same', use_bias=True, activation='relu')(y)
    y = tf.keras.layers.BatchNormalization(momentum=bnmomemtum)(y)
    y = fire_module(24, 48)(y)
    y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)
    y = fire_module(48, 96)(y)
    y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)
    y = fire_module(64, 128)(y)
    y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)
    y = fire_module(48, 96)(y)
    y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)
    y = fire_module(24, 48)(y)
    y = tf.keras.layers.GlobalAveragePooling2D()(y)
    outputs = layers.Dense(1, activation="sigmoid")(y)

    return tf.keras.Model(inputs, outputs)



def main():

    # ---------------------------------
    # PATH TO THE TRAINING FILES
    #path = "/Users/garrettchristian/DocumentsDesktop/uva21/classes/ml/project/"
    path = "/p/firedetection/dataML/"

    # ---------------------------------
    # GET THE TRAINING DATA

    batch_size = 32
    img_height = 254
    img_width = 254


    train_data_dir = path + 'Training/'

# https://stackoverflow.com/questions/42443936/keras-split-train-test-set-when-using-imagedatagenerator

    train_datagen = ImageDataGenerator(rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2) # set validation split


    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        classes = ['Fire', 'No_Fire'],
        class_mode='binary',
        subset='training',
        shuffle = True,
        seed=1337) # set as training data

    validation_generator = train_datagen.flow_from_directory(
        train_data_dir, # same directory as training data
        target_size=(img_height, img_width),
        batch_size=batch_size,
        classes = ['Fire', 'No_Fire'],
        class_mode='binary',
        subset='validation',
        shuffle = True,
        seed=1337) # set as validation data


    # ---------------------------------
    # SET UP THE MODEL

    model = getModel()
    model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss="binary_crossentropy",
              metrics=["accuracy"])

    print(model.summary())

    callbacks = [keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"), ]

    # ---------------------------------
    # TRAIN THE MODEL

    #history = model.fit(training_generator, validation_data=validation_generator, epochs=10)
    history = model.fit(train_generator, validation_data=validation_generator, epochs=50, callbacks=callbacks, use_multiprocessing=True, workers=6)

    # ---------------------------------
    # SAVE THE MODEL

    model.save("fireModel")


if __name__ == '__main__':
    main()

