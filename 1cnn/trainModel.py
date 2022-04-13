import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np


from PIL import Image
from PIL import ImageOps

from sklearn.model_selection import train_test_split

from tensorflow import keras

import glob

# Adapted from:
# https://www.tensorflow.org/tutorials/images/cnn

def getModel():

    model = models.Sequential()

    # Input
    # shape is Height x Width x Channels
    model.add(layers.Input(shape=(254, 254, 3)))

    # Convolutional Layers
    model.add(layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same')) 
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same')) 
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same')) 
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same')) 
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same')) 
    model.add(layers.MaxPooling2D((2, 2)))

    # Dense Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(1568, activation='relu'))
    model.add(layers.Dense(784, activation='relu'))
    model.add(layers.Dense(392, activation='relu'))
    model.add(layers.Dense(196, activation='relu'))
    model.add(layers.Dense(98, activation='relu'))
    
    # Output
    model.add(layers.Dense(2))

    return model


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

            # TODO need to do this correctly
            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)



def main():

    # ---------------------------------
    # PATH TO THE TRAINING FILES
    path = "/Users/garrettchristian/DocumentsDesktop/uva21/classes/ml/project/"
    # path = "/p/firedetection/dataML/"

    # ---------------------------------
    # GET THE TRAINING DATA

    fireTrain = np.array(glob.glob(path + "Training/Fire/*.jpg", recursive = True))
    print("Fire train shape", np.shape(fireTrain))
    nofireTrain = np.array(glob.glob(path + "Training/No_Fire/*.jpg", recursive = True))
    print("No fire train shape", np.shape(nofireTrain))

    

    files = np.concatenate((fireTrain, nofireTrain), axis=0)
    # labels = np.concatenate((labelsFire, labelsNofire), axis=0)
    print("Training Files Shape", np.shape(files))
    print("Training Labels Shape", np.shape(labels))

    labelsTmp = np.ones(np.shape(files)[0]) 

    train_data, test_data, _, _ = train_test_split(
        files, labelsTmp, test_size=0.2, random_state=21
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


    # Generators
    training_generator = DataGenerator(train_data, labels, **params)
    validation_generator = DataGenerator(test_data, labels, **params)

    # ---------------------------------
    # SET UP THE MODEL

    model = getModel()
    model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

    print(model.summary())

    # ---------------------------------
    # TRAIN THE MODEL

    history = model.fit(training_generator, validation_data=validation_generator, epochs=20)
    # history = model.fit(training_generator, validation_data=validation_generator, epochs=20, use_multiprocessing=True)

    # ---------------------------------
    # SAVE THE MODEL

    model.save("fireModel")


if __name__ == '__main__':
    main()

