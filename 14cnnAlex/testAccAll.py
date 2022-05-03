
import glob
import numpy as np
from PIL import Image
import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ---------------------------------
# PATH TO THE TEST FILES
#path = "/Users/garrettchristian/DocumentsDesktop/uva21/classes/ml/project/"
path = "/p/firedetection/dataML/"

test_data_dir = path + 'Test/'

batch_size = 256
img_height = 254
img_width = 254

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        test_data_dir, # same directory as training data
        target_size=(img_height, img_width),
        batch_size=batch_size,
        classes = ['Fire', 'No_Fire'],
        class_mode='binary',
        shuffle = True,
        seed=1337) 

 

# ---------------------------------
# LOAD THE MODEL
  
for i in range(1, 41):
        modelName = 'save_at_' + str(i) + '.h5'
        model = keras.models.load_model(modelName)
        # print("Info for ", modelName)

        # ---------------------------------
        # GET ACCURACY

        test_loss, test_acc = model.evaluate(test_generator, verbose=2)


        print("accuracy for ", i, test_acc)
