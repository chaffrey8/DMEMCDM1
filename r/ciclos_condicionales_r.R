# Declaración

# nombre_funcion <- function(argumentos){
  # Cuerpo función 
  # return()
#}

# Primero argumentos que no tienen valor por defecto
# Después los argumentos con defecto su sintáxis es arg=valor_defecto

suma_dos <- function(x,y){
  if(!is.numeric(x) & !is.numeric(y)){
    print('Los argumentos deben ser números.')
    return()
  }else if(!is.numeric(y)){
    print('El segundo argumento debe ser un número.')
    return()
  }else if(!is.numeric(x)){
    print('El primer argumento debe ser un número.')
    return()
  }else{
    return(x+y)
  }
}

suma_dos_plus <- function(x,y){
  if(!is.numeric(x) & !is.numeric(y)){
    print('Los argumentos deben ser números.')
    return()
  }
  if(!is.numeric(y)){
    print('El segundo argumento debe ser un número.')
    return()
  }
  if(!is.numeric(x)){
    print('El primer argumento debe ser un número.')
    return()
  }
  return(x+y)
}

casos_prueba <- list(list('gato',3),list(3,'gato'),list(3,2),list('gato','perro'))
resultados <- list()

for(i in 1:4){
  x <- casos_prueba[[i]][1]
  y <- casos_prueba[[i]][2]
  resultado_dos <- suma_dos(x,y)
  resultado_plus <- suma_dos_plus(x,y)
  if(!is.null(resultado_dos)&!is.null(resultado_plus)){
    if(resultado_dos==resultado_plus){
      resultados[i] <- TRUE
    }else{
      resultados[i] <- FALSE
    }
  }
  if(is.null(resultado_dos)&is.null(resultado_plus)){
    resultados[i] <- TRUE
  }else{
    resultados[i] <- FALSE
  }
}

print('')
print(resultados)
print(sum(resultados==TRUE))

resultados <- unlist(resultados)
ix <- 1
while(ix <= length(resultados)){
  if(resultados[ix]){
    resultados[ix] <- 1
  }else{
    resultados[ix] <- 0
  }
  ix <- ix +1
}

print('')
print(resultados)
print(sum(resultados))
