import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import train_test_split
from pydotplus import graph_from_dot_data

# Cargar los archivos que están separados por entrenamiento y test
train_df = pd.read_csv('Archivos/titanic-train.csv')
test_df = pd.read_csv('Archivos/titanic-test.csv')

# Mostrar información del Dataframe de Entrenamiento
print(train_df.info())

hombres = train_df.loc[train_df['Sex'] == 'male']  # Separar todos los Hombres
mujeres = train_df.loc[train_df['Sex'] == 'female']  # Separar todas las Mujeres

# Graficar sobrevivientes según el sexo y la clase
fig_1 = sns.catplot(x='Sex', y='Survived', col='Pclass',
                    kind='bar', data=train_df, saturation=.8, aspect=.6)
fig_1.set_axis_labels('', 'Tasa de Supervivencia')
fig_1.set_xticklabels(['Hombre', 'Mujer'])
fig_1.set_titles(col_template='{col_name} clase')
plt.show()

# Graficar sobrevivientes según Número de Padres o Hijos y la clase
fig_2 = sns.catplot(x='Parch', y='Survived', col='Pclass',
                    kind='bar', data=train_df, saturation=.8, aspect=.6, ci=None)
fig_2.set_axis_labels('', 'Tasa de Supervivencia')
fig_2.set_titles(col_template='{col_name} clase')
plt.show()

# Completar datos Nulos
print('\nColumnas con datos nulos:')
print(train_df.isnull().any())  # Mirar que columnas tienen datos nulos

# Las columnas que tienen datos nulos son: Age, Cabin, Embarked

print('\nCuantos datos nulos tiene cada columna:')
print(train_df.isnull().sum())  # Cuenta cuantos datos nulos tiene cada columna

# Corregir datos nulos de la columna Age rellenando datos con la media
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].mean())  # Mediana: median()

# Corregir datos nulos de Embarked (Puerto de embarque)
print('\nPersonas que embarcaron en: \nCherbourg: {} \nQueenstown: {} \nSouthampton: {}'
      .format((train_df.loc[train_df['Embarked'] == 'C']).shape[0],
              (train_df.loc[train_df['Embarked'] == 'Q']).shape[0],
              (train_df.loc[train_df['Embarked'] == 'S']).shape[0]))

train_df['Embarked'] = train_df['Embarked'].fillna('S')  # Southampton donde más personas embarcaron

# Eliminar 'Survived' (Porque es lo que se va a predecir) y otras etiquetas innecesarias para el entrenamiento
train_df_modificado = train_df.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)

# Separar los datos Numéricos y Categóricos
variables_num = train_df_modificado.select_dtypes(include='number').columns.tolist()

variables_categ = (train_df_modificado.select_dtypes(
    include='object').nunique() < 10)  # Número de elementos distintos menor a 10 como factor de seguridad
variables_categ = variables_categ[variables_categ == True].index.tolist()  # Convertor el objeto tipo serie

# Separar los datos o variables
x = pd.concat(
    [pd.get_dummies(train_df_modificado[variables_categ], drop_first=True), train_df_modificado[variables_num]],
    axis=1)  # Concatenar los datos numéricos y Categóricos ya codificados
y = train_df.Survived

# Dividir los datos para Entrenamiento y Test
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Creación del Modelo
modelo_AD = tree.DecisionTreeClassifier(max_leaf_nodes=10)  # max_leaf_nodes=10: Número máximo de Nodos - Hojas

# Entrenar el modelo
modelo_AD.fit(X_train, Y_train)

# Predecir con los datos de X_test
y_pred = modelo_AD.predict(X_test)

# Evaluar el rendimiento con datos de Test
print('\nPorcentaje de Rendimiento: {:.2f}%'.format(modelo_AD.score(X_test, Y_test) * 100))

# Mostrar Gráficamente los grad_fahr
dot_data = tree.export_graphviz(modelo_AD,
                                feature_names=X_train.columns.values,
                                filled=True,
                                rounded=True,
                                class_names=['No sobrevive', 'Sobrevive'])
graph = graph_from_dot_data(dot_data)
graph.write_png('árbol.png')

# Matriz de confusión
cfc_matrix = metrics.confusion_matrix(Y_test, y_pred)

# Graficar Matriz de confusión
res = ['Sobrevivieron', 'No Sobrevivieron']

grafica = metrics.plot_confusion_matrix(modelo_AD, X_test, Y_test, display_labels=res, cmap='Blues_r')
grafica.figure_.suptitle('Matriz de Confusión')
grafica.ax_.set_xlabel('Valor Predicho')
grafica.ax_.set_ylabel('Valor Verdadero')
