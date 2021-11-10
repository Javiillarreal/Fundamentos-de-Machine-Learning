import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # Dividir los datos
from sklearn.linear_model import LinearRegression

# Cargar los datos
dataset = pd.read_csv('Archivos/salarios.csv')

# Cuantos datos tiene
print('Tamaño del dataset', dataset.shape)

# Separar los datos o variables
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Dividir los datos para entrenamiento
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=3)

# Creación del modelo
regressor = LinearRegression()

# Entrenar el modelo
regressor.fit(X_train, Y_train)

# Mostrar gráficamente el resultado del entrenamiento
plt.scatter(X_train, Y_train)  # scatter -> Diagrama de dispersión
plt.plot(X_train, regressor.predict(X_train), color='black')  # función predict
plt.title('Salario VS Experiencia de Programadores')
plt.xlabel('Experiencia')
plt.ylabel('Salario')
plt.show()

# Mostrar gráficamente la validación del entrenamiento
plt.scatter(X_test, Y_test, color='red')  # scatter -> Diagrama de dispersión
plt.plot(X_train, regressor.predict(X_train), color='black')  # función predict
plt.title('Salario VS Experiencia de Programadores')
plt.xlabel('Experiencia')
plt.ylabel('Salario')
plt.show()

# Porcentaje de acierto
print(regressor.score(X_test, Y_test))
