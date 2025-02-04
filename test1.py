import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import keras

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.applications.vgg19 import VGG19, preprocess_input , decode_predictions


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import matplotlib.pyplot as plt
import numpy as np
import os


train_datagen = ImageDataGenerator(zoom_range= 0.5, shear_range= 0.3, horizontal_flip= True, preprocessing_function= preprocess_input)

val_datagen = ImageDataGenerator(preprocessing_function= preprocess_input)

train = train_datagen.flow_from_directory(directory= r"D:\finalyearproject\Dataset\train", target_size= (256,256), batch_size=32)

val = val_datagen.flow_from_directory(directory= r"D:\finalyearproject\Dataset\valid", target_size= (256,256), batch_size=32)

t_img , label = train.next()

def plotImage(img_arr, label):
  for im , l in zip(img_arr, label):
    plt.figure(figsize=(5,5))
    plt.imshow(im/255)
    plt.show()





from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.vgg19 import VGG19
import keras

base_model = VGG19(input_shape=(256,256,3),include_top= False)

for layer in base_model.layers:
   layer.trainable = False



X = Flatten()(base_model.output)

X = Dense(units= 10, activation='softmax')(X)

model = Model(base_model.input, X)



model.compile(optimizer= 'adam' , loss= keras.losses.categorical_crossentropy , metrics=['accuracy'])


# load best model

from keras.models import load_model
model = load_model(r"D:\finalyearproject\best_model.h5")





ref = dict(zip(list(train.class_indices.values()), list(train.class_indices.keys())))


def prediction(path):
  img = load_img(path, target_size= (256,256))

  i = img_to_array(img)

  im = preprocess_input(i)
  img = np.expand_dims(im , axis=0)
  pred = np.argmax(model.predict(img))
  print(f" the image belongs to {ref[pred] }")


path = r"D:\PlantDieaseDetectioncppproject\Dataset\test\Tomato__late_blight.JPG"
prediction(path)


