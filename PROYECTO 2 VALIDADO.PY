import math as mt

def biseccion(a, b, f, precision):
    diferencia = 1
    while diferencia > precision:
        c = (a + b) / 2
        fa = f(a)
        fc = f(c)
        if fa * fc < 0:
            b = c
        else:
            fb = f(b)
            if fc * fb < 0:
                a = c
        diferencia = abs(a - b)
    return (a + b) / 2  # Retorna el punto medio del último intervalo

def busca_intervalo(a, b, f):
    fa = f(a)
    fb = f(b)
    return fa * fb < 0

def main():
    raices = []
    
    # Solicitar la función al usuario
    while True:
        func_str = input("\nIngrese la función f(x) usando 'x' como variable y 'mt' para funciones matemáticas (ej. mt.sin(x)): ")
        try:
            # Definir la función dinámica
            def f(x):
                return eval(func_str, {'mt': mt, 'x': x})
            # Probar la función en un valor
            f(0)  # Evaluar en un punto para verificar que no haya errores
            break
        except Exception as e:
            print(f"Error en la función ingresada: {e}. Intente de nuevo.")

    # Solicitar porcentaje de error y calcular precisión
    while True:
        error_input = input("\nIngrese el porcentaje de error deseado: ")
        try:
            error_percent = float(error_input)
            if error_percent <= 0:
                raise ValueError("El porcentaje de error debe ser mayor que 0.")
            precision = error_percent
            break
        except ValueError:
            print("Entrada inválida. Asegúrese de ingresar un número válido.")

    # Solicitar intervalo
    while True:
        try:
            a_input = input("\nIngresa el límite inferior del intervalo: ")
            b_input = input("\nIngresa el límite superior del intervalo: ")
            a = float(a_input)
            b = float(b_input)
            if a >= b:
                raise ValueError("El límite inferior debe ser menor que el límite superior.")
            break
        except ValueError:
            print("Entrada inválida. Asegúrese de ingresar números válidos.")

    paso = 0.1  # Tamaño del paso para dividir el intervalo
    
    # Verificar raíces en los límites
    if f(a) == 0:
        raices.append(a)
    if f(b) == 0:
        raices.append(b)
    
    i = a + paso
    
    # Buscar raíces en subintervalos
    while i <= b:
        print(f"Buscando en el intervalo... ({a:.3f}, {i:.3f})")
        if busca_intervalo(a, i, f):
            print("\t¡Hay una raíz! Aplicando bisección...")
            raiz = biseccion(a, i, f, precision)
            print("*******************************************")
            print(f"\tRaíz encontrada: {raiz:.6f}")
            print("*******************************************")
            raices.append(raiz)
        a = round(i, 3)
        i = round(a + paso, 3)
    
    # Mostrar resultados
    print("Raíces encontradas:", [round(r, 6) for r in raices])

if __name__ == '__main__':
    main()