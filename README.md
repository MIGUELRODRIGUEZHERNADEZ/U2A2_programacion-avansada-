# U2A2_programacion-avansada-
PROGRAMA. Implementación de Perceptrón
import numpy as np

# Datos de la tabla (puntaje de crédito, ingresos, monto del préstamo, relación deuda/ingresos)
X = np.array([
    [750, 5.0, 20.0, 0.3],
    [600, 3.0, 15.0, 0.6],
    [680, 4.0, 10.0, 0.4],
    [550, 2.5, 8.0, 0.7],
    [800, 6.0, 25.0, 0.2]
])

# Salida esperada (Solicitud Aprobada: 1 = Sí, 0 = No)
y = np.array([1, 0, 1, 0, 1])

# Normalización de datos (min-max scaling)
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# Parámetros del perceptrón
learning_rate = 0.1
epochs = 20
weights = np.random.rand(4)  # 4 características
bias = np.random.rand()

def activation_function(x):
    return 1 if x >= 0 else 0

# Entrenamiento del perceptrón
for epoch in range(epochs):
    print(f"Época {epoch + 1}:")
    for i in range(len(X)):
        linear_output = np.dot(X[i], weights) + bias
        prediction = activation_function(linear_output)
        error = y[i] - prediction

        weights += learning_rate * error * X[i]
        bias += learning_rate * error

        print(f"  Muestra {i+1}: Entrada {X[i]}, Salida esperada {y[i]}, Predicción {prediction}, Error {error}")
    print(f"  Pesos actualizados: {weights}, Bias actualizado: {bias}\n")

# Evaluación del modelo con un nuevo dato
nuevo_dato = np.array([700, 4.5, 12.0, 0.5])  # Nuevo dato de prueba
nuevo_dato = (nuevo_dato - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))  # Normalización
salida = activation_function(np.dot(nuevo_dato, weights) + bias)
print(f"El modelo evalúa la solicitud {nuevo_dato} como: {'Aprobada' if salida == 1 else 'Rechazada'}")
