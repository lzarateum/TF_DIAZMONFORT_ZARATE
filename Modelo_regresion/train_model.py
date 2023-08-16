import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
import joblib

# Paso 2: Leer el archivo CSV
data = pd.read_csv('medidas.csv', sep='\,')

# Paso 3: Separar características y columna de salida
X = data.drop(['NAME','EYE', 'NASAL_SIDE'], axis=1)
y = data['NASAL_SIDE'].replace({'R': 0, 'L': 1}) 



# Paso 4: Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.astype('float64')
y_train = y_train.astype('float64')
X_test = X_test.astype('float64')
y_test = y_test.astype('float64')
#Paso 5: construir el modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01), input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Capa de salida con activación sigmoid para clasificación binaria
])
#Paso 5: compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10000, batch_size=32, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Puedes usar el modelo entrenado para hacer predicciones en nuevos datos
new_data = pd.read_csv('test.csv')  # Carga nuevos datos desde un archivo CSV similar
new_pred = new_data.drop(['NAME','EYE','NASAL_SIDE'], axis=1)
predictions = model.predict(new_pred)

for i in range(len(new_pred)):
    print(new_data['NAME'][i], new_data['EYE'][i])
    print(f"Sample {i+1} - Score R: {predictions[i][0]}, Score L: {1 - predictions[i][0]}")


#joblib.dump(model, 'vault_model_v1.pkl')
