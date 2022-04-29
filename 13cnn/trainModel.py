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

import glob

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Adapted from:
# https://www.tensorflow.org/tutorials/images/cnn
# https://colab.research.google.com/drive/1nseete5huZlWM7Ak0qL-T75Dbk0mdr-Z?usp=sharing#scrollTo=o-qUPyfO7Qr8
# https://github.com/AlirezaShamsoshoara/Fire-Detection-UAV-Aerial-Image-Classification-Segmentation-UnmannedAerialVehicle/blob/30ae2526a5e0e744701f24bb44a2de62e355aece/training.py


def getModel():

    inputs = keras.Input(shape=(254, 254, 3))
    # x = data_augmentation(inputs)  # 1) First option
    x = inputs  # 2) Second option

    # x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    # x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.Conv2D(8, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x

    # for size in [128, 256, 512, 728]:
    for size in [8]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)

        x = layers.add([x, residual])
        previous_block_activation = x
    x = layers.SeparableConv2D(8, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)


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
    history = model.fit(train_generator, validation_data=validation_generator, epochs=40, callbacks=callbacks, use_multiprocessing=True, workers=6)

    # ---------------------------------
    # SAVE THE MODEL

    model.save("fireModel")


if __name__ == '__main__':
    main()

