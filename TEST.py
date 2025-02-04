from flask import Flask
from flask import Flask, render_template, request
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




def prediction(tomato_plant):
  img = load_img(tomato_plant, target_size= (256,256))

  i = img_to_array(img)

  im = preprocess_input(i)
  img = np.expand_dims(im , axis=0)
  result = np.argmax(model.predict(img))

  
  print('@@ Raw result = ', result)
  if result==0:
      return "Apple - Apple Scab Disease", 'Apple_Scab.html'
       
  elif result==1:
      return "Apple - Black Rot Disease", 'Apple_Black_rot.html'
        
  elif result==2:
      return "Apple - Cedar Rust", 'Apple_Cedar_apple_rust.html'
        
  elif result==3:
      return "Apple - Apple Healthy", 'Applehealthy.html'
       
  elif result==4:
      return "Blueberry - Healthy", 'Blueberry_healthy.html'
        
  elif result==5:
      return "Cherry - Healthy", 'Cherry_healthy.html'
        
  elif result==6:
      return "Cherry - Powdery Mild", 'Cherry_powdery_mild.html'
        
  elif result==7:
      return "Corn - Leaf Spot", 'Corn_cercospora_leaf_spot.html'
    
  elif result==8:
      return "Corn - Common Rust", 'Corn_common_rust.html'
        
  elif result==9:
      return "Corn - Healthy", 'Corn_healthy.html'

  elif result==10:
      return "Corn - Northern Leaf Blight", 'Corn_northern_leaf_blight.html'

  elif result==11:
      return "Grape - Black Rot", 'Grape_Black_rot.html'

  elif result==12:
      return "Grape - Esca Black Mealeses", 'Grape_Esca_Black_mealeses.html'

  elif result==13:
      return "Grape - Healthy", 'Grape_healthy.html'

  elif result==14:
      return "Grape - Leaf Blight", 'Grape_leaf_blight.html'

  elif result==15:
      return "Orange - Haunglongbing", 'Orange_honglongbling.html'

  elif result==16:
      return "Peach - Bacterial Spot", 'Peach_bacterial_spot.html'

  elif result==17:
      return "Peach - Healthy", 'Peach_healthy.html'

  elif result==18:
      return "Peeper - Bacterial Spot", 'Peeper_bell_bacterial_spot.html'

  elif result==19:
      return "Peeper - Healthy", 'Peeper_bell_healthy.html'

  elif result==20:
      return "Potato - Early Blight", 'Potato_earlyblight.html'

  elif result==21:
      return "Potato - Healthy", 'Potato_healthy.html'

  elif result==22:
      return "Potato - Late Blight", 'Potato_lateblight.html'

  elif result==23:
      return "Raspberry - Healthy", 'Raspberry_healthy.html'

  elif result==24:
      return "Soyabean - Healthy", 'Soyabean_healthy.html'

  elif result==25:
      return "Squash - Powdery Mild", 'squash_powderly_mild.html'

  elif result==26:
      return "Strawberry - Healthy", 'Strawberry_healthy.html'

  elif result==27:
      return "Strawberry - Leaf Scorch", 'Strawberry_leaf_soarch.html'

  elif result==28:
      return "Tomato - Bacteria Spot Disease", 'Tomato-Bacteria Spot.html'

  elif result==29:
      return "Tomato - Early Blight Disease", 'Tomato-Early_Blight.html'

  elif result==30:
      return "Tomato - Healthy and Fresh", 'Tomato-Healthy.html'

  elif result==31:
      return "Tomato - Late Blight Disease", 'Tomato - Late_blight.html'

  elif result==32:
      return "Tomato - Leaf Mold Disease", 'Tomato - Leaf_Mold.html'

  elif result==33:
      return "Tomato - Septoria Leaf Spot Disease", 'Tomato - Septoria_leaf_spot.html'

  elif result==34:
      return "Tomato - Two Spotted Spider Mite Disease", 'Tomato - Two-spotted_spider_mite.html'

  elif result==35:
      return "Tomato - Target Spot Disease", 'Tomato - Target_Spot.html'

  elif result==36:
      return "Tomato - Tomato Mosaic Virus Disease", 'Tomato - Tomato_mosaic_virus.html'

  elif result==37:
      return "Tomato - Tomoato Yellow Leaf Curl Virus Disease", 'Tomato - Tomato_Yellow_Leaf_Curl_Virus.html'

   


# Create flask instance
app = Flask(__name__)


 #render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
        return render_template('index.html')
    
 
# get input image from client then predict class and render respective .html page for solution
@app.route("/predict", methods = ['GET','POST'])
def predict():
     if request.method == 'POST':
        file = request.files['image'] # fet input
        filename = file.filename        
        print("@@ Input posted = ", filename)
        
        file_path = os.path.join(r'D:\finalyearproject\static', filename)
        file.save(file_path)

        print("@@ Predicting class......")
        pred, output_page = prediction(tomato_plant=file_path)
              
        return render_template(output_page, pred_output = pred, user_image = file_path)

if __name__ == "__main__":
    app.run(threaded=False,port=8080) 

