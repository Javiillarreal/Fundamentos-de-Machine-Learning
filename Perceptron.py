import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Fórmula Grados Celcius a fahrenheit
def cel2fahr(c):
    f = c * 1.8 + 32
    return f


print(cel2fahr(100))

# Generar datos
grad_cel = np.random.randint(-10, 10, 10)
print('Grados Celcius:\n', grad_cel)

grad_fahr = np.round(list(map(cel2fahr, grad_cel)), 2)
print('\nGrados Fahrenhet:\n', grad_fahr)

# Aprendizaje automático
capa = tf.keras.layers.Dense(units=1,
                             input_shape=[1])  # Input_shape[1] indica que hay una capa de entrada con una neurona
oculta_1 = tf.keras.layers.Dense(5)
oculta_2 = tf.keras.layers.Dense(5)
salida=tf.keras.layers.Dense(1)
# modelo = tf.keras.Sequential([capa])
modelo = tf.keras.Sequential([capa, oculta_1, oculta_2, salida])

# Compilas el modelo
modelo.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error')

# Entrenar el modelo
historial = modelo.fit(grad_cel, grad_fahr, epochs=100, verbose=False)

# Graficar la función de pérdida
plt.figure()
plt.xlabel('Épocas')
plt.ylabel('Magnitud de Pérdida')
plt.plot(historial.history['loss'])

# Predicción
celcius = 100
resultado = modelo.predict([celcius])
print('El valor en grados Fahrenheit de:', celcius, 'grados Celcius es:', resultado[0, 0], 'grados Fahrenheit')

# Pesos y Sesgos asignados
print('Pesos y Sesgos asignados:\n', capa.get_weights())
print('Pesos y Sesgos asignados:\n', oculta_1.get_weights())
