# -*- coding: utf-8 -*-
"""img_test_mkx a csv.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bV1hC7c74zKed4VGqiV97u_ieY683sxR
"""



import pandas as pd
import numpy as np
import tensorflow as tf
import json
from os.path import isfile, join
from os import listdir

from google.colab import drive
drive.mount('/content/drive')

directorio = ("/content/drive/MyDrive/TRABAJO FINAL/img_test_mkx/")

#leemos imagenes dentro de la carpeta

files = [f for f in listdir(directorio) if isfile(join(directorio, f))]

name = []
eye = []
ata = []
nIRISW = []
tIRISW = []
n_iris3_l =[]
t_iris3_l = []
nasal_side = []

for fname in files:
    f = join(directorio, fname)

    with open(f, 'br') as file:
        data = file.read()
        data = json.loads(data)
    # Sacamos parámetros de interés
    try:
      name.append(data['extra_info']['name'])
      eye.append(data['extra_info']['eye'])
      ata.append(data['lines']['ATA']['value']['length'].replace('mm','').strip())
      nIRISW.append(data['lines']['nIRISW']['value']['length'].replace('mm','').strip())
      tIRISW.append(data['lines']['tIRISW']['value']['length'].replace('mm','').strip())
      n_iris3_l.append(data['p3s']['n-iris3']['value']['length'])
      t_iris3_l.append(data['p3s']['t-iris3']['value']['length'])



    # Indicamos lado nasal y temporal
      if (data['angles']['n-ang-post']['x0'] < data['angles']['t-ang-post']['x0']):
          sideL = 'N'
          sideR = 'T'
      else:
          sideL = 'T'
          sideR = 'N'


      if sideL == 'N':
          nasal_side.append('L')
      else:
          nasal_side.append('R')

    except:
      print(fname)

data = {
    'NAME': name,
    'EYE': eye,
    'AtA': ata,
    'NASAL IRIS WIDTH': nIRISW,
    'TEMPORAL IRIS WIDTH': tIRISW,
    'NASAL IRIS LARGE': n_iris3_l,
    'NASAL SIDE': nasal_side,
    'TEMPORAL IRIS LARGE': t_iris3_l,

}

# Convertir los datos en un DataFrame de pandas
df = pd.DataFrame(data, columns=['NAME', 'EYE', 'AtA', 'NASAL IRIS WIDTH', 'TEMPORAL IRIS WIDTH', 'NASAL IRIS LARGE', 'NASAL SIDE', 'TEMPORAL IRIS LARGE'])

# Guardar el DataFrame en un archivo CSV
df.to_csv('/content/drive/MyDrive/TRABAJO FINAL/img_test.csv', index=False)

