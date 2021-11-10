import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split  # Dividir los datos
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

# Cargar los datos
dataset = pd.read_csv('Archivos/salarios_pais.csv')
print('\nTamaño del Dataset: ', len(dataset))

# Crear vector de Países
paises = ['CO', 'MX', 'US', 'DE']

# Países a Números
le = preprocessing.LabelEncoder()
pais2num = le.fit_transform(paises)  # Codificar a números
print('\nPaís:', le.inverse_transform(pais2num), '\nCódigo:', pais2num, '\n')

# Mostrar a que país corresponde cada número
dic_paises = dict(zip(paises, pais2num))
for key in dic_paises:
    print(key, ':', dic_paises[key])

# 30 nuevos elementos aleatorios par el Dataset
new_colum = np.zeros((len(dataset)), dtype=int)

for i in range(len(dataset)):
    new_colum[i] = np.random.choice(pais2num)

# Agregar paises codificados a Dataset
dataset['Pais'] = new_colum

# Separar los datos o variables
x = dataset.drop('Salario', axis=1)  # Elimina la columna Salario
y = dataset.iloc[:, 1].values  # Columna Salario

# Dividir los datos para Entrenamiento y Test
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Creación del modelo
regressor = LinearRegression()

# Entrenar el modelo
regressor.fit(X_train, Y_train)

# Predicción o Salida del modelo
y_pred = regressor.predict(X_test)

# Mostrar Predicción con datos de Test
print('\n             Predicción\n{:15} {:10} {:10}'.format('Experiencia', 'País', 'Salario'))
for i in range(len(X_test)):
    print('{:^14} {} {:^18.0f}'.format(X_test.iloc[i, 0], le.inverse_transform([X_test.iloc[i, 1]]), y_pred[i]))

# Evaluar el rendimiento con datos de Test
print('\nRendimiento: ', regressor.score(X_test, Y_test))

# Mostrar gráficamente los grad_fahr
fig = plt.figure(dpi=150)
grafica = fig.add_subplot(111, projection='3d')
grafica.scatter(X_train['Aexperiencia'], X_train['Pais'], Y_train, color='blue', label='Train')
grafica.scatter(X_test['Aexperiencia'], X_test['Pais'], Y_test, color='red', marker='s', label='Test')
grafica.plot_trisurf(X_train['Aexperiencia'], X_train['Pais'], regressor.predict(X_train), color='black',
                     alpha=0.3)
grafica.set_title('Salario Experiencia y pais')
grafica.set_xlabel('Experiencia')
grafica.set_ylabel('Pais')
grafica.set_zlabel('Salario')
grafica.set_yticks(range(len(pais2num)))
grafica.set_yticklabels(le.inverse_transform(pais2num))
plt.legend(loc='upper left')

fig.show()
