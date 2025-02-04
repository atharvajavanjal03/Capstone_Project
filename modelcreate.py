

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

train = train_datagen.flow_from_directory(directory= "/content/planthealthdetection/Dataset/train", target_size= (256,256), batch_size=32)

val = val_datagen.flow_from_directory(directory= "/content/planthealthdetection/Dataset/val", target_size= (256,256), batch_size=32)

t_img , label = train.next()

def plotImage(img_arr, label):
  for im , l in zip(img_arr, label):
    plt.figure(figsize=(5,5))
    plt.imshow(im/255)
    plt.show()

plotImage(t_img[:3], label[:3])

from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.vgg19 import VGG19
import keras

base_model = VGG19(input_shape=(256,256,3),include_top= False)

for layer in base_model.layers:
   layer.trainable = False

base_model.summary()

X = Flatten()(base_model.output)

X = Dense(units= 10, activation='softmax')(X)


#creating our model
model = Model(base_model.input, X)

model.summary()

model.compile(optimizer= 'adam' , loss= keras.losses.categorical_crossentropy , metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint, EarlyStopping

# early stopping
es = EarlyStopping(monitor= 'val_accuracy', min_delta= 0.01, patience= 3, verbose=1)

# model check point
mc = ModelCheckpoint(filepath="best_model.h5", 
                     monitor= 'val_accuracy', 
                     min_delta= 0.01, 
                     patience= 3, 
                     verbose=1, 
                     save_best_only= True )

cb = [es,mc]

his = model.fit_generator(train, 
                          steps_per_epoch= 10, 
                          epochs= 50, 
                          verbose= 1, 
                          callbacks=cb , 
                          validation_data= val , 
                          validation_steps=10 )

h = his.history
h.keys()

plt.plot(h['accuracy'])
plt.plot(h['val_accuracy'] , c = "red")
plt.title("acc vs v-acc")
plt.show()

plt.plot(h['loss'])
plt.plot(h['val_loss'] , c = "red")
plt.title("loss vs v-loss")
plt.show()

# load best model

from keras.models import load_model
model = load_model("/content/best_model.h5")

acc = model.evaluate_generator(val)[1]

print(f"The accuracy of your model is {acc*100} %")

ref = dict(zip(list(train.class_indices.values()), list(train.class_indices.keys())))

def prediction(path):
  img = load_img(path, target_size= (256,256))

  i = img_to_array(img)

  im = preprocess_input(i)
  img = np.expand_dims(im , axis=0)
  pred = np.argmax(model.predict(img))
  print(f" the image belongs to {ref[pred] }")

path = "/content/planthealthdetection/Dataset/test/Tomato___Tomato_Yellow_Leaf_Curl_Virus (1).JPG"
prediction(path)
