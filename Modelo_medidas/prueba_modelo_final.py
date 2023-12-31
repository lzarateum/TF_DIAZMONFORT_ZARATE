# -*- coding: utf-8 -*-
"""prueba modelo final.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TJM-PS80NEFYPRiZ7BeYEQFUP4h3qDKa
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from keras.utils import to_categorical
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv

from google.colab import drive
drive.mount('/content/drive')

model = tf.keras.models.load_model("/content/drive/MyDrive/TRABAJO FINAL/LeftRight_model_230718.keras")

m_ata = 11.528129411764706
s_ata = 0.8437984954357611
m_niW = 0.5055823529411765
s_niW = 0.152288606561953
m_tiW = 0.4805882352941176
s_tiW = 0.14557999976706942
m_niL = 3.9707776960502503
s_niL = 0.43708067525763855
m_tiL = 5.2074269027821805
s_tiL = 0.7214663629935241

f = pd.read_csv("/content/drive/MyDrive/TRABAJO FINAL/img_test.csv")
f.head()

result = []

for index, row in f.iterrows():
  ata = (row['AtA']-m_ata)/s_ata
  niW = (row['NASAL IRIS WIDTH']-m_niW)/s_niW
  tiW = (row['TEMPORAL IRIS WIDTH']-m_tiW)/s_tiW
  niL = (row['NASAL IRIS LARGE']-m_niL)/s_niL
  tiL = (row['TEMPORAL IRIS LARGE']-m_tiL)/s_tiL

  #print(index)
  data_in = np.array ([niL, tiL]).reshape(1, 2)
  #print(data_in)
  prediccion = model.predict(data_in, verbose=0)
  p = round(prediccion[0][0])
  print(p)
  if p == 0:
    print('izquierdo')
  else:
    print('derecho')
  ######################

  nSide_real = row['NASAL SIDE']
  if nSide_real == 'L':
    nSide_real = 0
  else:
    nSide_real = 1

  result.append(nSide_real-p)


print(result)

#ata, niW, tiW,

#Para un solo archivo mkx
#ata = data['lines']['ATA']['value']['length'].replace('mm','').strip()
#nIRISW = data['lines']['nIRISW']['value']['length'].replace('mm','').strip()
#tIRISW = data['lines']['tIRISW']['value']['length'].replace('mm','').strip()
#n_iris3_l = data['p3s']['n-iris3']['value']['length']
#t_iris3_l = data['p3s']['t-iris3']['value']['length']

#normal_ata_train = (float(ata)-m_ata)/s_ata
#normal_niW_train = (float(nIRISW)-m_niW)/s_niW
#normal_tiW_train = (float(tIRISW)-m_tiW)/s_tiW
#normal_niL_train = (float(n_iris3_l)-m_niL)/s_niL
#normal_tiL_train = (float(t_iris3_l)-m_tiL)/s_tiL

#target_test = to_categorical(target_test)

