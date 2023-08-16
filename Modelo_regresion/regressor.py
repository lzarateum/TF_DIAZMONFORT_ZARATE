import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import joblib

# Paso 2: Leer el archivo CSV
data = pd.read_csv('medidas.csv', sep='\,')

# Paso 3: Separar características y columna de salida
X = data.drop(['NAME','EYE','NASAL_SIDE','N_IRIS_LARGE'], axis=1)
y = data['N_IRIS_LARGE']


# Paso 5: Normalizar los datos de entrada
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X)



# Paso 6: Crear y entrenar la red neuronal
model = MLPRegressor(hidden_layer_sizes=(50,50), max_iter=10000)
model.fit(X_train_norm, y)

# Paso 7: Realizar predicciones
data = pd.read_csv('test.csv', sep='\,')
X_test = data.drop(['NAME','EYE','NASAL_SIDE', 'N_IRIS_LARGE'], axis=1)
X_test_norm = scaler.transform(X_test)
y_pred = model.predict(X_test_norm)

print(y_pred/data['T_IRIS_LARGE'])
print(data['N_IRIS_LARGE']/data['T_IRIS_LARGE'])
print(y_pred)
joblib.dump(model, 'regres_model_v1.pkl')
