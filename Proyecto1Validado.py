import numpy as np

def gauss_seidel(A, B, tolerance=1e-10, max_iterations=1000):
    """
    Resuelve un sistema de ecuaciones lineales usando el método de Gauss-Seidel.
    
    Parámetros:
        A (numpy.ndarray): Matriz de coeficientes (n x n)
        B (numpy.ndarray): Vector de términos independientes (n x 1)
        tolerance (float): Error máximo permitido (por defecto 1e-10)
        max_iterations (int): Número máximo de iteraciones (por defecto 1000)
    
    Retorna:
        numpy.ndarray: Vector solución x
    """
    n = A.shape[0]
    
    # Validaciones iniciales
    if A.shape[1] != n:
        raise ValueError("La matriz A debe ser cuadrada")
    if B.shape[0] != n:
        raise ValueError("Las dimensiones de A y B no coinciden")
    
    # Verificar si la matriz es diagonal dominante
    is_diagonally_dominant = True
    for i in range(n):
        diagonal = abs(A[i, i])
        row_sum = np.sum(np.abs(A[i, :])) - diagonal
        if diagonal <= row_sum:
            is_diagonally_dominant = False
    
    if not is_diagonally_dominant:
        print("Advertencia: La matriz no es estrictamente diagonal dominante. La convergencia no está garantizada.")
    
    x = np.zeros_like(B, dtype=np.float64)
    
    for iteration in range(max_iterations):
        x_prev = x.copy()
        
        # Iteración de Gauss-Seidel
        for i in range(n):
            sigma = 0.0
            for j in range(n):
                if j != i:
                    sigma += A[i, j] * x[j]
            x[i] = (B[i] - sigma) / A[i, i]
        
        # Calcular error relativo máximo
        error = np.max(np.abs((x - x_prev) / (np.abs(x) + 1e-12)))  # Evitar división por cero
        
        if error < tolerance:
            print(f"Convergencia alcanzada en {iteration + 1} iteraciones")
            return x
    
    print(f"Máximo de iteraciones alcanzado ({max_iterations})")
    return x

def leer_matriz(n):
    """Función para leer y validar la matriz A ingresada por el usuario."""
    A = []
    for i in range(n):
        while True:
            try:
                row = list(map(float, input(f"Fila {i+1}: ").strip().split()))
                if len(row) != n:
                    raise ValueError(f"La fila debe contener {n} elementos")
                A.append(row)
                break
            except ValueError:
                print("Error: Ingrese números válidos separados por espacios.")
    return np.array(A)

def leer_vector(n):
    """Función para leer y validar el vector B ingresado por el usuario."""
    while True:
        try:
            B = list(map(float, input("Ingrese el vector B: ").strip().split()))
            if len(B) != n:
                raise ValueError(f"El vector debe contener {n} elementos")
            return np.array(B)
        except ValueError:
            print("Error: Ingrese números válidos separados por espacios.")

def leer_parametros():
    """Función para leer y validar la tolerancia y el número máximo de iteraciones."""
    while True:
        try:
            tolerance = float(input("Tolerancia (por defecto 1e-10): ") or 1e-10)
            max_iter = int(input("Iteraciones máximas (por defecto 1000): ") or 1000)
            return tolerance, max_iter
        except ValueError:
            print("Error: Ingrese valores numéricos válidos.")

if __name__ == "__main__":
    while True:
        try:
            n = int(input("Tamaño de la matriz (n): "))
            if n <= 0:
                raise ValueError("El tamaño de la matriz debe ser un número positivo.")
            break
        except ValueError:
            print("Error: Ingrese un número entero positivo.")
    
    print("\nIngrese la matriz A fila por fila (valores separados por espacios):")
    A = leer_matriz(n)
    
    print("\nIngrese el vector B:")
    B = leer_vector(n)
    
    tolerance, max_iter = leer_parametros()
    
    # Resolver sistema
    solution = gauss_seidel(A, B, tolerance, max_iter)
    
    # Mostrar resultados
    print("\nSolución encontrada:")
    print(solution)
