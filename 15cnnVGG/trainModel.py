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


def getModel():

    # Adapted from:
    # https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c
    # VGG net

    model = Sequential()
    model.add(layers.Conv2D(input_shape=(254, 254, 3),filters=64, kernel_size=(3,3),padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(units=4096,activation="relu"))
    model.add(layers.Dense(units=4096,activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    return model



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

