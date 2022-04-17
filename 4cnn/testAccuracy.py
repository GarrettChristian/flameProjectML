
import glob
import numpy as np
from PIL import Image
import keras

from trainModel import DataGenerator

# ---------------------------------
# PATH TO THE TEST FILES
#path = "/Users/garrettchristian/DocumentsDesktop/uva21/classes/ml/project/"
path = "/p/firedetection/dataML/"

fireTest = np.array(glob.glob(path + "Test/Fire/*.jpg", recursive = True))
print("Fire test shape", np.shape(fireTest))
nofireTest = np.array(glob.glob(path + "Test/No_Fire/*.jpg", recursive = True))
print("No fire test shape", np.shape(nofireTest))

labels = {}

for fireFile in fireTest:
    labels[fireFile] = 1

for nofireFile in nofireTest:
    labels[nofireFile] = 0


filesTest = np.concatenate((fireTest, nofireTest), axis=0)
print("Files Test Shape", np.shape(filesTest))


params = {'dim': (254, 254),
            'batch_size': 32,
            'n_classes': 2,
            'n_channels': 3,
            'shuffle': True}

eval_generator = DataGenerator(filesTest, labels, **params)
 

# ---------------------------------
# LOAD THE MODEL
  
modelName = 'fireModel'
model = keras.models.load_model(modelName)
print("Info for ", modelName)

# ---------------------------------
# GET ACCURACY

test_loss, test_acc = model.evaluate(eval_generator, verbose=2)

print(test_acc)
