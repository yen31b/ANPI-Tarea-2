import numpy as np
import math
from time import time
from multiprocessing import Process
import os
import winsound

def parte1_p3():
    # Se definen los parámetros
    x0 = np.zeros(242)
    p = q = np.arange(1, 25.2, 0.1)
    m = 242
    b = np.ones(m)
    tol = 1e-5
    iterMax = 1000
    A = tridiagonal(p, q, m)
    # Llamada al método de Jacobi con los parámetros definidos
    #metodo_jacobi_paralelo(A, b, x0, tol, iterMax)

    if __name__ == '__main__':
        # Se empieza a contar el tiempo
        tiempo_inicial = time()
        #lista de procesos
        procesos = []
        #cantidad de cores a utilizar
        #cores = os.cpu_count()
        cores = 2
        print('Cantidad de cores: ', cores)

        # Crear procesos
        print('Creando procesos...')
        for n in range(cores):
            proceso = Process(target=metodo_jacobi_paralelo, args=(A, b, x0, tol, iterMax))
            procesos.append(proceso)
        # Ejecutar los procesos
        print('Ejecutando procesos...')
        for proceso in procesos:
            proceso.start()    
        
        # Espera de la ejecucion de los procesos
        for proceso in procesos:
            proceso.join()
        # Parar el tiempo
        tiempo_final = time() - tiempo_inicial

        print(f'Tiempo total en segundos: {tiempo_final}s\n')

def tridiagonal(p, q, m):
    # Crear la matriz A
    A = np.zeros((m, m))
    # Definir los valores que no varían en la matriz
    A[0, 0] = 2 * q[0]
    A[0, 1] = q[0]
    A[1, 0] = p[1]
    A[1, 2] = q[1]
    # Definir los valores que varían por el índice
    for i in range(1, m-1):
        A[i, i-1] = p[i]    # diagonal inferior
        A[i, i] = 2 * (p[i] + q[i])  # diagonal principal
        A[i, i+1] = q[i]    # diagonal superior
    # Definir valores finales de la matriz
    A[m-1, m-2] = p[m-1]
    A[m-1, m-1] = 2 * p[m-1]
    return A

def metodo_jacobi_paralelo(A, b, x0, tol, iterMax):
    # Obtener elementos de la diagonal
    d = np.diag(A)
    # Dimensión de b para saber cuál debe ser la dimensión de aproximación xk
    m = len(b)
    # Inicializar xk
    xk = x0.copy()
    # Calcular la aproximación por medio de la fórmula
    for k in range(iterMax):
        xk1 = np.zeros(m)
        for i in range(m):
            sumatoria = 0
            for j in range(m):
                if j != i:
                    sumatoria += A[i, j] * xk[j]
            xk1[i] = (b[i] - sumatoria) / A[i, i]
        criterio_parada = np.linalg.norm(A @ xk1 - b)
        if criterio_parada < tol:            
            #print(f'Resultado final xk: {xk}\n')
            print(f'El criterio de parada fue: {criterio_parada} \t')
            print(f'Convergió en {k+1} iteraciones \t')
            return xk1
        xk = xk1
    print('El método de Jacobi no convergió en el número máximo de iteraciones')
    return xk

# Llamar a la función principal
parte1_p3()