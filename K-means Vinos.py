import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn import metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Cargar los datos
df_vinos = datasets.load_wine()

# Normalizar los datos de X (ya que cada feature tiene diferentes rangos)
scaler = StandardScaler()
scaler.fit(df_vinos.data)
x_scaler = scaler.transform(df_vinos.data)

# Separar los datos o variables
X = pd.DataFrame(scaler.transform(df_vinos.data), columns=df_vinos.feature_names)
y = df_vinos.target  # Solo se utiliza para evaluar ya que es NO supervisado

# Ver si el Dataset tiene datos nulos
print('\nCuantos datos nulos tiene cada columna:')
print(X.isnull().sum())  # Cuenta cuantos datos nulos tiene cada columna

# Encontrar el Número de clusters por medio del método del codo
wcss = []

for k in range(1, 11):
    k_means = KMeans(n_clusters=k, random_state=0)
    k_means.fit(X)
    wcss.append(k_means.inertia_)  # devuelve la suma de las distancias al cuadrado de las muestras a su centro

# Graficar la suma de las distancias
plt.plot(range(1, 11), wcss, 'go-', linewidth=1, markersize=8)
plt.title('Método del Codo')
plt.xlabel('Número de Clusters')
plt.ylabel('WCSS')
plt.show()

# Creación del modelo
K_vinos = KMeans(n_clusters=3, max_iter=1000)

# Entrenar el modelo
K_vinos.fit(X)

# Etiquetas encontradas basadas en similitud
y_labels = K_vinos.labels_

# Predicción
y_predic = K_vinos.predict(X)

# Evaluar el Rendimiento
print('Porcentaje de Rendimiento: {:.2f}%'.format(metrics.adjusted_rand_score(y, y_predic) * 100))

