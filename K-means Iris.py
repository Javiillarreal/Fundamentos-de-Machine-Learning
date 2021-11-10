from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Cargar los datos
iris = datasets.load_iris()

# Separar los Datos o variables
x = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target  # Solo se utiliza para evaluar ya que es NO supervisado

# Mostrar Gráficamente los Datos
fig_1 = plt.figure(dpi=100)

ax1 = fig_1.add_subplot(121)
ax1.scatter(x[iris.feature_names[2]], x[iris.feature_names[3]], color='blue', edgecolors='black')
ax1.set_title('Longitud VS Ancho\ndel Pétalo')
ax1.set_xlabel('Longitud del Pétalo', fontsize=10)
ax1.set_ylabel('Ancho del Pétalo', fontsize=10)

ax2 = fig_1.add_subplot(122)
ax2.scatter(x[iris.feature_names[0]], x[iris.feature_names[1]], color='red', edgecolors='black')
ax2.set_title('Longitud VS Ancho\ndel Sépalo')
ax2.set_xlabel('Longitud del Sépalo', fontsize=10)
ax2.set_ylabel('Ancho del Sépalo', fontsize=10)

# Creación del modelo
model_K = KMeans(n_clusters=3, max_iter=1000)  # Número de K o centroides a generar y maxímo de iteraciones

# Entrenar el modelo
model_K.fit(x)

# Etiquetas encontradas basadas en la similitud
y_labels = model_K.labels_

# Predicción (Cómo ha dividido)
y_predic = model_K.predict(x)

# Convertir a DF la prediccion para plotear con seaborn
df_y_predic = pd.DataFrame(y_predic, columns=['Clase'])

# Evaluar el rendimiento
print('Porcentaje de Rendimiento: {:.2f}%'.format(
    metrics.adjusted_rand_score(y, y_predic) * 100))  # Compara aleatoriamente sin importar el orden

# Mostrar Gráficamente los grad_fahr
fig_2 = plt.figure(dpi=100)

ax1 = fig_2.add_subplot(121)
ax1.scatter(x[iris.feature_names[2]], x[iris.feature_names[3]], c=[y_predic], s=30, edgecolors='black')
ax1.set_title('Longitud VS Ancho\ndel Pétalo')
ax1.set_xlabel('Longitud del Pétalo', fontsize=10)
ax1.set_ylabel('Ancho del Pétalo', fontsize=10)

ax2 = fig_2.add_subplot(122)
ax2.scatter(x[iris.feature_names[0]], x[iris.feature_names[1]], c=[y_predic], edgecolors='black')
ax2.set_title('Longitud VS Ancho\ndel Sépalo')
ax2.set_xlabel('Longitud del Sépalo', fontsize=10)
ax2.set_ylabel('Ancho del Sépalo', fontsize=10)

# Graficar Pares de conjuntos de las variables
Z = pd.concat([pd.DataFrame(iris.data, columns=iris.feature_names), df_y_predic], axis=1)
sns.pairplot(Z, hue='Clase', markers=["o", "s", "D"], palette='Dark2') #diag_kind="none"
plt.show()
