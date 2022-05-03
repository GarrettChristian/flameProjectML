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

    # adapted from
    # https://towardsdatascience.com/implementing-alexnet-cnn-architecture-using-tensorflow-2-0-and-keras-2113e090ad98
    # based on alexnet

    inputs = layers.Input(shape=(254, 254, 3))

    x = inputs
    x = layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=(3,3), strides=(2,2))(x)
    x = layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=(3,3), strides=(2,2))(x)
    x = layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=(3,3), strides=(2,2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation='relu')(x)
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
    history = model.fit(train_generator, validation_data=validation_generator, epochs=50, callbacks=callbacks, use_multiprocessing=True, workers=6)

    # ---------------------------------
    # SAVE THE MODEL

    model.save("fireModel")


if __name__ == '__main__':
    main()

