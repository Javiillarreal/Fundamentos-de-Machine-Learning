import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# Cargar los datos
dataset = pd.read_csv('Archivos/diabetes.csv')
print('\nTamaño del Dataset:', dataset.shape)

# Separar los datos o variables
x = dataset.drop('Outcome', axis=1)
y = dataset.Outcome  # Otra forma de seleccionar la columna de grad_fahr

# Dividir los datos para Entrenamiento y Test
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Creación del Modelo
log_reg = LogisticRegression(solver='lbfgs', max_iter=1000)  # Corrige un error de convergencia en las iteraciones

# Entrenar el Modelo
log_reg.fit(X_train, Y_train)

# Predecir con los datos de X_test
y_pred = log_reg.predict(X_test)

# Matriz de confusión
cfc_matrix = metrics.confusion_matrix(Y_test, y_pred)

# Graficar Matriz de Confusión
diabetes = ['Con Diabetes', 'Sin Diabetes']

grafica = metrics.plot_confusion_matrix(log_reg, X_test, Y_test, display_labels=diabetes, cmap='Blues_r')
grafica.figure_.suptitle('Matriz de Confusión')
grafica.ax_.set_xlabel('Valor Predicho')
grafica.ax_.set_ylabel('Valor Verdadero')

# Exactitud
print('\nExactitud:', metrics.accuracy_score(Y_test, y_pred))
