from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Cargar los Modelos Pre-entrenados
modelo_NN = keras.models.load_model('Clasificador')
modelo_CNN_Plus = keras.models.load_model('Clasificador_CNN_Aumento_Dropout')

# Cargar Dataset
fashion_Df = keras.datasets.fashion_mnist

# Separar los Datos de Entrenamiento y Prueba
(train_img, train_label), (test_img, test_label) = fashion_Df.load_data()

# Expandir dimensión de canal para CNN
test_img = np.expand_dims(test_img, axis=-1)

# Nombres de las Clases del Set de Datos ya que no están incluidas en el Dataset
class_names = ['Camiseta', 'Pantalón', 'Suéter', 'Vestido', 'Abrigo', 'Sandalia', 'Camisa', 'Zapatilla deportiva',
               'Bolso', 'Botines']

# Predicciones con los datos de Test utilizando los dos Modelos
predictions_NN = modelo_NN.predict(test_img)
predictions_CNN_Plus = modelo_CNN_Plus.predict(test_img)


# Graficar
def graficar_resultados(predictions, predictions_mod, titulo):
    # Mostrar Imagen
    def plot_img(i, predic_array, true_label, img):
        predic_array, true_label, img = predic_array, true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img, cmap='binary')

        predic_label = np.argmax(predic_array)

        if predic_label == true_label:
            color = 'green'
        else:
            color = 'red'

        plt.xlabel(
            '{} {:.2f}%\n({})'.format(class_names[predic_label], np.max(predic_array) * 100, class_names[true_label]),
            color=color)

    # Mostrar Gráfica
    def plot_porcentaje(i, predic_array, true_label):
        predic_array, true_label = predic_array, true_label[i]
        plt.grid(False)
        plt.xticks(range(10))
        plt.yticks([])
        porcentaje = plt.bar(range(10), predic_array, color='blue')
        plt.ylim([0, 1])
        predic_label = np.argmax(predic_array)

        porcentaje[predic_label].set_color('red')
        porcentaje[true_label].set_color('green')

    # Dividir Subplot en dos para graficar datos Originales y Modificados
    fig = plt.figure(figsize=(3.5 * num_columnas, 2.5 * num_filas), dpi=75)

    cont_0 = 0
    cont_1 = 0
    cont_2 = 0

    for fila in range(num_filas):
        for columna in range(num_columnas):
            if columna < 2:

                fig.add_subplot(num_filas, 2 * num_columnas, 2 * cont_0 + 1)
                plot_img(cont_1, predictions[cont_1], test_label, test_img)

                fig.add_subplot(num_filas, 2 * num_columnas, 2 * cont_0 + 2)
                plot_porcentaje(cont_1, predictions[cont_1], test_label)

                cont_1 += 1

            else:

                for X, Y in datagen.flow(test_img, test_label, batch_size=num_img, shuffle=False):

                    fig.add_subplot(num_filas, 2 * num_columnas, 2 * cont_0 + 1)
                    plot_img(cont_2, predictions_mod[cont_2], test_label, X)

                    fig.add_subplot(num_filas, 2 * num_columnas, 2 * cont_0 + 2)
                    plot_porcentaje(cont_2, predictions_mod[cont_2], test_label)

                    break

                cont_2 += 1
            cont_0 += 1

    plt.text(11.8, 2.0,
             '0 Camiseta\n1 Pantalón\n2 Suéter\n3 Vestido\n4 Abrigo\n5 Sandalia\n6 Camisa\n7 Zapatilla deportiva\n8 '
             'Bolso\n9 Botines', bbox={'facecolor': 'gray', 'alpha': 0.5, 'pad': 5})

    plt.suptitle(
        '         {}\n          Imágenes Originales              Imágenes Modificadas'.format(titulo), fontsize=23)

    plt.tight_layout()
    plt.show()


# Generar Imágenes modificadas
rango_rotacion = 50
mov_ancho = 1.5
mov_alto = 1.5
rango_acercamiento = [0.9, 1.1]

datagen = ImageDataGenerator(
    rotation_range=rango_rotacion,
    width_shift_range=mov_ancho,
    height_shift_range=mov_alto,
    zoom_range=rango_acercamiento)

datagen.fit(test_img)

# # Predicciones Nuevos Datos Generados
num_filas = 5
num_columnas = 4

num_img = (num_filas * num_columnas)

for X, Y in datagen.flow(test_img, test_label, batch_size=test_img.shape[0], shuffle=False):
    predictions_NN_Mod = modelo_NN.predict(X)
    predictions_CNN_Plus_Mod = modelo_CNN_Plus.predict(X)
    break

# Predicciones modelo NN
graficar_resultados(predictions_NN, predictions_NN_Mod, 'Modelo NN')

# Predicciones modelo CNN Plus
graficar_resultados(predictions_CNN_Plus, predictions_CNN_Plus_Mod, 'Modelo CNN Plus')
