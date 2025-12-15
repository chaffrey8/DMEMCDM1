# Declaración

# def nombre_funcion(argumentos):
  # Cuerpo función 
  # return 

# Primero argumentos que no tienen valor por defecto
# Después los argumentos con defecto su sintáxis es arg=valor_defecto
# python si puede hacer validación de typo desde su declaración
# También se puede definir el tipo de objeto que regresa
from typeguard import typechecked

@typechecked
def suma_dos(x:int|float,y:int|float) -> int | float:
    return x + y

def suma_dos_plus(x,y) -> int | float | None:
    if not isinstance(x,(int,float)) and not isinstance(y,(int,float)):
        print('Los argumentos deben ser números.')
        return None
    elif not isinstance(y,(int,float)):
        print('El segundo argumento debe ser un número.')
        return None
    elif not isinstance(x,(int,float)):
        print('El primer argumento debe ser un número.')
        return None
    else:
        return x+y

casos_prueba = [('gato',3),(3,'gato'),(3,2),('gato','perro')]
resultados = [False]*len(casos_prueba)

for i in range(4):
    x = casos_prueba[i][0]
    y = casos_prueba[i][1]
    resultado_plus = suma_dos_plus(x,y)
    try:
       resultado_dos = suma_dos(x,y)
    except:
       resultado_dos = None
    finally:
       if resultado_dos == resultado_plus:
           resultados[i] = True

print('\n')
print(resultados)
print(sum(resultados))

ix = 0
while True:
    if ix >= len(resultados):
        break
    resultados[ix] *= 1
    ix += 1

print('\n')
print(resultados)
print(sum(resultados))

casos_prueba = [('gato',3),(3,'gato'),(3,2),('gato','perro')]
resultados = [False]*len(casos_prueba)

for i,caso in enumerate(casos_prueba):
    x = caso[0]
    y = caso[1]
    resultado_plus = suma_dos_plus(x,y)
    try:
       resultado_dos = suma_dos(x,y)
    except:
       resultado_dos = None
    finally:
       if resultado_dos == resultado_plus:
           resultados[i] = True

print('\n')
print(resultados)
print(sum(resultados))