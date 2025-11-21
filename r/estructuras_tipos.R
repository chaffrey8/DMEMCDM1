# Vectores

x <- c('banana','manzana','naranja','fresa')
print(x)

## Operaciones
mi_vector <- c(10, 20, 30, 40, 50)
mi_vector + 5         # Suma 5 a cada elemento del vector
mi_vector * 2         # Multiplica cada elemento por 2
sum(mi_vector)        # Suma de todos los elementos
mean(mi_vector)       # Promedio de los elementos
mi_vector[1]          # Acceso al primer elemento
mi_vector[2:4]        # Acceso a un rango de elementos
mi_vector[1] = 60     # Cambio de un elemento
mi_vector[seq(1,5,2)] = c(10,50,30)   # Cambio de varios elementos
mi_vector[3:5] <- c(30,40,50)
mi_vector[mi_vector > 25]  # Filtrado de elementos mayores a 25
otro_vector <- c(5,3,6,7,8)
mi_vector + otro_vector # Si tienen la misma longitud las operaciones son entrada a entrada
tercer_vector <- c(1,2,3,2,1,2,3,2,1,2)
mi_vector + tercer_vector # Si un vector tienen longitudes proporcionales entonces repite el de menor longitud las veces necesarias para operar entrada a entrada

# Matrices
matrix(1:6,nrow=2,ncol=3)
matrix(1:6,nrow=2,ncol=3,byrow = TRUE)print(x)

# Operaciones comunes
mi_matriz <- matrix(1:9, nrow = 3, ncol = 3)
mi_matriz * 2                 # Multiplicación de todos los elementos por 2
mi_matriz + 10                # Suma de 10 a cada elemento
mi_matriz[1, 2]               # Acceso al elemento en la primera fila, segunda columna
mi_matriz[ , 1]               # Acceso a la primera columna completa
rowSums(mi_matriz)            # Suma de cada fila
colMeans(mi_matriz)           # Promedio de cada columna
t(mi_matriz)                  # Transposición de la matriz
otra_matriz <- matrix(1:15, nrow=5, ncol=3)
otra_matriz%*%mi_matriz       # Producto matricial

# Listas
mi_lista <- list(nombre = "Ana", edad = 25, notas = c(8, 9, 10))

# Operaciones comunes
mi_lista$nombre               # Acceso a un elemento por nombre
mi_lista[["edad"]]            # Acceso a un elemento por nombre (otra forma)
mi_lista[[3]][2]              # Acceso al segundo elemento dentro del tercer elemento (notas)
mi_lista$edad <- 26           # Modificar un elemento de la lista
mi_lista$nuevo_elemento <- "Nuevo"  # Agregar un nuevo elemento a la lista
length(mi_lista)              # Número de elementos en la lista


# Data Frames
mi_df <- data.frame(nombre = c("Ana", "Luis", "María"), edad = c(25, 30, 22), notas = c(8, 9, 7))

# Operaciones comunes
mi_df$edad                   # Acceso a una columna completa
mi_df[1, "nombre"]           # Acceso a un elemento específico
mi_df[mi_df$edad > 25, ]     # Filtrar filas donde la edad es mayor a 25
mi_df$notas <- mi_df$notas * 10  # Operar sobre una columna específica
mi_df$nueva_col <- c("A", "B", "C")  # Agregar una nueva columna
nrow(mi_df)                  # Número de filas
ncol(mi_df)                  # Número de columnas
summary(mi_df)              # Resumen estadístico del data frame


# Factores
mi_factor <- factor(c("alto", "medio", "bajo", "medio", "alto"))

# Operaciones comunes
levels(mi_factor)            # Ver los niveles del factor
table(mi_factor)             # Frecuencia de cada nivel
mi_factor <- factor(mi_factor, levels = c("bajo", "medio", "alto"))  # Cambiar el orden de los niveles
mi_factor[2]                 # Acceso a un elemento
mi_factor[mi_factor == "alto"] <- "muy alto"  # Error porque no se ha agregado el nuevo nivel
levels(mi_factor) <- c(levels(mi_factor), "muy alto")     # Agregar un nuevo nivel al factor
mi_factor[mi_factor == "alto"] <- "muy alto"    # Modificar nivel en un elemento

# Creación de un array de 3 dimensiones
mi_array <- array(1:12, dim = c(3, 2, 2))

# Operaciones comunes
mi_array[1, 2, 1]           # Acceso a un elemento específico
mi_array[, , 1]             # Acceso a la primera "capa" del array
apply(mi_array, c(1, 2), sum)  # Suma de los elementos en la tercera dimensión
mi_array * 2                # Multiplicación de todos los elementos por 2
dim(mi_array)               # Dimensiones del array
