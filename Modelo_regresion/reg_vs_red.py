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

data = pd.read_csv('test.csv', sep='\,')
X_test = data.drop(['NAME','EYE','NASAL_SIDE', 'N_IRIS_LARGE'], axis=1)
n_real = data['N_IRIS_LARGE']

scaler = StandardScaler()
X_test_norm = scaler.fit_transform(X_test)

model = joblib.load('regres_model_v1.pkl')
n_pred = model.predict(X_test_norm)

t_large = np.array(X_test['T_IRIS_LARGE'])
n_rl = t_large*0.6395256849 + 0.69594020914

e_pred = sum(abs(n_real - n_pred))/len(n_real)
e_reg = sum(abs(n_real - n_rl))/len(n_real)
print(e_pred)
print(e_reg)
    
#print(abs(y_real - y_pred))
#print(abs(y_real - n_large))
plt.scatter(t_large, n_real, label='Real')
plt.scatter(t_large, n_rl, label='Regresión Lineal')
plt.scatter(t_large, n_pred, label='Predicción IA')
plt.legend()
plt.grid(True)
plt.show()


