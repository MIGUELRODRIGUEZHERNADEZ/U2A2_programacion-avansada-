## programacion avansada 
# 18 de febrero del 2025
# Elaborado por Miguel Angel Rodriguez Hernadez
# introduccion 
En el sector financiero, la evaluación de solicitudes de préstamo es un proceso 
crítico para minimizar riesgos y optimizar la aprobación de créditos. 
Tradicionalmente, esta evaluación ha sido realizada por analistas financieros, pero 
con el avance de la inteligencia artificial, se han implementado modelos de 
aprendizaje supervisado para automatizar esta tarea. En este informe, se presenta 
la implementación y el entrenamiento de un perceptrón en Python para clasificar 
solicitudes de préstamo en aprobadas (1) o rechazadas (0), en función de variables 
financieras relevantes. 

## Problema a Resolver 
Una institución financiera desea automatizar la clasificación de solicitudes de 
préstamo mediante un perceptrón que evalúe los siguientes factores clave: 

 Puntaje de crédito: Valor numérico entre 300 y 850. 

 Ingresos mensuales: Expresado en miles de pesos. 

 Monto del préstamo solicitado: Expresado en miles de pesos. 

 Relación deuda/ingresos: Valor decimal (por ejemplo, 0.2, 0.5, etc.). 

La institución proporciona un conjunto de datos históricos con ejemplos de 
solicitudes aprobadas y rechazadas. El perceptrón debe aprender a clasificar 
correctamente cada solicitud. 


``` import numpy as np

X = np.array([
    [750, 5.0, 20.0, 0.3],
    [600, 3.0, 15.0, 0.6],
    [680, 4.0, 10.0, 0.4],
    [550, 2.5, 8.0, 0.7],
    [800, 6.0, 25.0, 0.2]
])
y = np.array([1, 0, 1, 0, 1])

X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

learning_rate = 0.1
epochs = 20
weights = np.random.rand(4)  # 4 características
bias = np.random.rand()

def activation_function(x):
    return 1 if x >= 0 else 0

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

nuevo_dato = np.array([700, 4.5, 12.0, 0.5])  # Nuevo dato de prueba
nuevo_dato = (nuevo_dato - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))  # Normalización
salida = activation_function(np.dot(nuevo_dato, weights) + bias)
print(f"El modelo evalúa la solicitud {nuevo_dato} como: {'Aprobada' if salida == 1 else 'Rechazada'}")
