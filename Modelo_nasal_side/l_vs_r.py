import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.layers.regularization.dropout import Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import requests
from io import BytesIO
import cv2

#Crear el dataset generador
datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range = 10,
    shear_range = 5,
    validation_split=0.2 #20% para pruebas
)

#Generadores para sets de entrenamiento y pruebas
data_gen_entrenamiento = datagen.flow_from_directory('./dataset',
                                                     target_size=(224, 224),
                                                     batch_size=32,
                                                     shuffle=True,
                                                     subset='training')
data_gen_pruebas = datagen.flow_from_directory('./dataset',
                                               target_size=(224, 224),
                                               batch_size=32,
                                               shuffle=True,
                                               subset='validation')
print(data_gen_entrenamiento.class_indices)

url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
mobilenetv2 = hub.KerasLayer(url, input_shape=(224,224,3))
mobilenetv2.trainable = False

modelo = tf.keras.Sequential([
    mobilenetv2,
    tf.keras.layers.Dense(2, activation='softmax'),
    tf.keras.layers.Dropout(0.1)
])
modelo.summary()

#Compilar como siempre
modelo.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

EPOCAS = 45
historial = modelo.fit(
    data_gen_entrenamiento, epochs=EPOCAS, batch_size=32,
    validation_data=data_gen_pruebas
)

def categorizar(imagen):
  img = Image.open(imagen)
  img = np.array(img).astype(float)/255
  plt.imshow(img)
  plt.show()
  img = cv2.resize(img, (224,224))
  print (img.shape)
  prediccion = modelo.predict(img.reshape(-1, 224, 224, 3))
  print (prediccion)
  return np.argmax(prediccion[0], axis=-1)
  
  
directory = './test/'
files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith('.jpg')]
files.sort()

for f_name in files:
   
    imagen = directory + f_name
    prediccion = categorizar (imagen)
    if prediccion == 0:
      print ('NASAL DERECHO')
    if prediccion == 1:
       print ('NASAL IZQUIERDO') 
    

#modelo.save('./model_nivsnd_231508_0.h5')
 
