# Cargar función read_csv de la librería readr
library(readr)

# Cargar librería dplyr para manejo de datos
library(dplyr)

# Ruta donde se encuentra el archivo
file_path <- '~/desktop/Diplomado/bases/adult/adult.data'

# Leer base de datos
df <- read_csv(
  file_path,
  col_names = c('age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income'),
  trim_ws = TRUE
)

# Pregunta 1
# ¿Cuántas mujeres menores de 35 años viven en Japón? 11
df |> filter(age <35 & sex == 'Female' & `native-country` == 'Japan') |> count()

# Pregunta 2
# ¿De los hombres divorciados que se dedican al tech-support, qué país de origen es el más común? Ninguno
df |> filter(sex == 'Male' & `marital-status` == 'Divorced' & occupation == 'Tech-support') |>  group_by(`native-country`) |> summarise(count = n()) |> arrange(desc(count))

# Pregunta 3
# Cuál es la edad promedio de las personas que viven en Irlanda o Francia que nunca han trabajado? Ninguna
df |> filter(`native-country` %in% c('Ireland','France') & workclass == 'Never-worked')