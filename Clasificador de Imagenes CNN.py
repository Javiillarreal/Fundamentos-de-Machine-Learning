from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

# Cargar los Datos
fashion_df = keras.datasets.fashion_mnist

# Separar los Datos de Entrenamiento y Prueba
(train_img, train_label), (test_img, test_label) = fashion_df.load_data()

# Nombres de las Clases del Set de Datos ya que no están incluidas en el Dataset
class_names = ['Camiseta', 'Pantalón', 'Suéter', 'Vestido', 'Abrigo', 'Sandalia', 'Camisa', 'Zapatilla deportiva',
               'Bolso', 'Botines']

# Pre-procesado de Datos
print('Escala:', train_img[0].min(), '-', train_img[0].max(), '\nTipo:', train_img[0].dtype)

train_img = train_img / 255  # Normalizar datos
test_img = test_img / 255
print('\nNueva Escala:', train_img[5654].min(), '-', train_img[5654].max(), '\nNuevo Tipo:', train_img[5654].dtype)

# Verificar el Formato adecuado y mostrarr las 25 primeras Imágenes
plt.figure(dpi=75, figsize=(10, 10))

for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_img[i], cmap='binary')
    plt.xlabel(class_names[train_label[i]])

# Expandir dimensión de canal para CNN
train_img = np.expand_dims(train_img, axis=-1)
test_img = np.expand_dims(test_img, axis=-1)

# Aumento de Datos
rango_rotacion = 50
mov_ancho = 1.5
mov_alto = 1.5
# rango_inclinacion = 5
rango_acercamiento = [0.9, 1.1]

datagen = ImageDataGenerator(
    rotation_range=rango_rotacion,
    width_shift_range=mov_ancho,
    height_shift_range=mov_alto,
    zoom_range=rango_acercamiento,
    # shear_range=rango_inclinacion
)

datagen.fit(train_img)

# Nuevos Datos de Entrenamiento
train_data_gen = datagen.flow(train_img, train_label)

# Codigo para mostrar imagenes del set, imprime como se ven antes y despues de las transformaciones
filas = 4
columnas = 8
num = filas * columnas
print('ANTES:\n')
fig1, axes1 = plt.subplots(filas, columnas, figsize=(1.5 * columnas, 2 * filas))
for i in range(num):
    ax = axes1[i // columnas, i % columnas]
    ax.imshow(train_img[i].reshape(28, 28), cmap='gray_r')
    ax.set_title('{}'.format(class_names[train_label[i]]))
plt.tight_layout()
plt.show()
print('DESPUES:\n')
fig2, axes2 = plt.subplots(filas, columnas, figsize=(1.5 * columnas, 2 * filas))
for X, Y in datagen.flow(train_img, train_label, batch_size=num, shuffle=False):
    for i in range(0, num):
        ax = axes2[i // columnas, i % columnas]
        ax.imshow(X[i].reshape(28, 28), cmap='gray_r')
        ax.set_title('{}'.format(class_names[Y[i]]))
    break
plt.tight_layout()
plt.show()

# Definir el modelo de Red Neuronal
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'),  # Convolucion con kernel de 3,3
    keras.layers.MaxPooling2D(2, 2),  # Agrupación máxima con matriz de 2,2

    # keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # keras.layers.MaxPooling2D(2, 2),

    keras.layers.Dropout(0.3),

    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compilar el Modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el Modelo
model.fit(train_data_gen, epochs=10)

# Guardar el modelo
# model.save('Clasificador_CNN_Aumento_Dropout')

# Evaluar la Exactitud
train_loss, train_acc = model.evaluate(train_img, train_label, verbose=0)  # Con Datos de Entrenamiento
test_loss, test_acc = model.evaluate(test_img, test_label, verbose=0)  # Con Datos de Test

print('Exactitud con datos de Entrenamiento:', train_acc)
print('Exactitud con datos de Test', test_acc)

# Predicciones utilizando el modelo Entrenado
predictions = model.predict(test_img)


# Graficar los grad_fahr
def plot_img(i, predic_array, true_label, img):
    predic_array, true_label, img = predic_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img[:,:,0], cmap='binary')

    predic_label = np.argmax(predic_array)

    if predic_label == true_label:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel('{} {:.2f}%\n({})'.format(class_names[predic_label], np.max(predic_array) * 100, class_names[true_label]),
               color=color)


def plot_porcentaje(i, predic_array, true_label):
    predic_array, true_label = predic_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks(range(100))
    porcentaje = plt.bar(range(10), predic_array, color='blue')
    plt.ylim([0, 1])
    predic_label = np.argmax(predic_array)

    porcentaje[predic_label].set_color('red')
    porcentaje[true_label].set_color('green')


num_filas = 5
num_columnas = 4

num_img = num_filas * num_columnas
plt.figure(figsize=(4 * num_columnas, 2.5 * num_filas))

for i in range(num_img):
    plt.subplot(num_filas, 2 * num_columnas, 2 * i + 1)
    plot_img(i, predictions[i], test_label, test_img)

    plt.subplot(num_filas, 2 * num_columnas, 2 * i + 2)
    plot_porcentaje(i, predictions[i], test_label)

plt.text(11.8, 2.0,
         '0 Camiseta\n1 Pantalón\n2 Suéter\n3 Vestido\n4 Abrigo\n5 Sandalia\n6 Camisa\n7 Zapatilla deportiva\n8 Bolso\n9 Botines',
         bbox={'facecolor': 'gray', 'alpha': 0.5, 'pad': 5})

plt.tight_layout()
plt.show()

# Predicción sobre una sola imagen
img = test_img[586]
plt.figure()
plt.imshow(img)
print('Tamaño de la imagen:', img.shape)

img = np.expand_dims(img, 0)  # Aumentar la dimension de la imagen ya que Keras trabaja sobre bloques

predic_img = model.predict(img)  # Predecir
print('La imagen corresponde a:', class_names[np.argmax(predic_img)])
