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

# Adapted from:
# https://www.tensorflow.org/tutorials/images/cnn

def getModel1():

    model = models.Sequential()

    # Input
    # shape is Height x Width x Channels
    model.add(layers.Input(shape=(254, 254, 3)))

    # Convolutional Layers
    model.add(layers.Conv2D(filters=8, kernel_size=3, strides=2, activation='relu', padding='same')) 
    model.add(layers.BatchNormalization())
    #model.add(layers.LeakyReLU(alpha=0.2))
    #model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Dropout(0.2))

    for size in [8]:
        model.add(layers.SeparableConv2D(filters=size, kernel_size=3, strides=1, activation='relu', padding='same')) 
    #model.add(layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same')) 
        model.add(layers.BatchNormalization())
    #model.add(layers.LeakyReLU(alpha=0.2))
    #model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Dropout(0.2))
        
        model.add(layers.SeparableConv2D(filters=size, kernel_size=3, strides=1, activation='relu', padding='same')) 
    #model.add(layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same')) 
        model.add(layers.BatchNormalization())
    #model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.MaxPooling2D(3, strides=2))
    #model.add(layers.Dropout(0.2))
        
    
    model.add(layers.SeparableConv2D(filters=8, kernel_size=3, strides=1, activation='relu', padding='same')) 
    #model.add(layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same')) 
    model.add(layers.BatchNormalization())
    #model.add(layers.LeakyReLU(alpha=0.2))
    #model.add(layers.MaxPooling2D(3, strides=2))
    #model.add(layers.Dropout(0.2))


    #model.add(layers.Conv2D(filters=16, kernel_size=3, strides=1, activation='relu', padding='same')) 
    #model.add(layers.BatchNormalization())
    #model.add(layers.LeakyReLU(alpha=0.2))
    #model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Dropout(0.2))


    #model.add(layers.Conv2D(filters=8, kernel_size=3, strides=1, activation='relu', padding='same')) 
    #model.add(layers.BatchNormalization())
    #model.add(layers.LeakyReLU(alpha=0.2))
    #model.add(layers.MaxPooling2D((2, 2)))
   # model.add(layers.Dropout(0.2))

    #model.add(layers.Conv2D(filters=8, kernel_size=3, strides=1, activation='relu', padding='same')) 
    #model.add(layers.BatchNormalization())
    #model.add(layers.LeakyReLU(alpha=0.2))
    #model.add(layers.MaxPooling2D((2, 2)))

    #model.add(layers.Conv2D(filters=128, kernel_size=3, strides=1, activation='relu', padding='same')) 
    #model.add(layers.BatchNormalization())
    #model.add(layers.LeakyReLU(alpha=0.2))
    #model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.GlobalAveragePooling2D())

    # Dense Layers
    #model.add(layers.Flatten())
    #model.add(layers.Dense(63504, activation='relu'))
    #model.add(layers.Dropout(0.2))
    #model.add(layers.Dense(30752, activation='relu'))
    #model.add(layers.Dropout(0.2))
    #model.add(layers.Dense(14400, activation='relu'))
    #model.add(layers.Dropout(0.2))
    #model.add(layers.Dense(8192, activation='relu'))
    #model.add(layers.Dropout(0.2))
    #model.add(layers.Dense(4096, activation='relu'))
    #model.add(layers.Dropout(0.2))
    #model.add(layers.Dense(2048, activation='relu'))
    #model.add(layers.Dropout(0.2))
    #model.add(layers.Dense(1024, activation='relu'))
    #model.add(layers.Dropout(0.2))
    #model.add(layers.Dense(512, activation='relu'))
    #model.add(layers.Dropout(0.2))
    #model.add(layers.Dense(256, activation='relu'))
    #model.add(layers.Dropout(0.2))
    #model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    #model.add(layers.Dense(64, activation='relu'))

    # Output
    model.add(layers.Dense(1))

    return model


def getModel():

    inputs = keras.Input(shape=(254, 254, 3))
    # x = data_augmentation(inputs)  # 1) First option
    x = inputs  # 2) Second option

    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
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
    outputs = layers.Dense(1)(x)
    return keras.Model(inputs, outputs)

# Adapted from:
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, batch_size=32, dim=(254, 254), n_channels=3,
                 n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            image = Image.open(ID)
            # imageGrey = ImageOps.grayscale(image)
            # imageGreyArray = np.array(imageGrey)


            

            imageArray = np.array(image)
            imageArrayNorm = imageArray.astype('float32') / 255

                

            X[i,] = imageArrayNorm

            # Store class
            y[i] = self.labels[ID]


        #return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, y


def main():

    # ---------------------------------
    # PATH TO THE TRAINING FILES
    #path = "/Users/garrettchristian/DocumentsDesktop/uva21/classes/ml/project/"
    path = "/p/firedetection/dataML/"

    # ---------------------------------
    # GET THE TRAINING DATA

    fireTrain = np.array(glob.glob(path + "Training/Fire/*.jpg", recursive = True))
    print("Fire train shape", np.shape(fireTrain))
    nofireTrain = np.array(glob.glob(path + "Training/No_Fire/*.jpg", recursive = True))
    print("No fire train shape", np.shape(nofireTrain))

    labelsF = np.ones(np.shape(fireTrain)[0]) 
    labelsNf = np.zeros(np.shape(nofireTrain)[0]) 
    

    files = np.concatenate((fireTrain, nofireTrain), axis=0)
    labelsSplit = np.concatenate((labelsF, labelsNf), axis=0)
    print("Training Files Shape", np.shape(files))
    print("Training Labels Shape", np.shape(labelsSplit))


    train_data, test_data, _, _ = train_test_split(
        files, labelsSplit, test_size=0.2, random_state=21, stratify=labelsSplit
    )

    labels = {}

    for fireFile in fireTrain:
        labels[fireFile] = 1
    
    for nofireFile in nofireTrain:
        labels[nofireFile] = 0

    # ---------------------------------
    # SET UP THE DATAGENERATOR

    # Parameters
    params = {'dim': (254, 254),
            'batch_size': 32,
            'n_classes': 2,
            'n_channels': 3,
            'shuffle': True}


    print("Train ", np.shape(train_data))
    print("Test ", np.shape(test_data))

    # Generators
    training_generator = DataGenerator(train_data, labels, **params)
    validation_generator = DataGenerator(test_data, labels, **params)

    # ---------------------------------
    # SET UP THE MODEL

    model = getModel()
    model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])

    print(model.summary())

    # ---------------------------------
    # TRAIN THE MODEL

    #history = model.fit(training_generator, validation_data=validation_generator, epochs=10)
    history = model.fit(training_generator, validation_data=validation_generator, epochs=40, use_multiprocessing=True, workers=6)

    # ---------------------------------
    # SAVE THE MODEL

    model.save("fireModel")


if __name__ == '__main__':
    main()

