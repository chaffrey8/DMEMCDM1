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

# Ver tibble
print(head(df))

# Filtrar
df_menores_edad <- df  |>
  filter(age < 18)

df_mujeres_menor_edad <- filter(df,age < 18 & sex == 'Female')

df_no_hombres_mayores <- filter(df,age < 18 | sex == 'Female')

# Ordenar
df_no_hombres_mayores |> arrange(age)
df_no_hombres_mayores |> arrange(desc(`education-num`))
df |> arrange(age,`education-num`)
df |> arrange(`education-num`,age)
df |> arrange(age,desc(`education-num`))
df |> arrange(`education-num`,desc(age))

# Mezclar
df1 <- df |> select(age,occupation) |> slice(1:10)
df2 <- df[seq(1,16,3),c('race','sex')]
df1 |> bind_rows(df2)
df1 |> bind_cols(df2) #! Error, ver nrow(df1) y nrow(df2)
df1 |> left_join(df2) #! Error, necesitan llave
df1 |> mutate(index = row_number()) |> left_join(df2 |> mutate(index = row_number()))
df1 <- df1 |> mutate(index = row_number())
df2 <- df2 |> mutate(index = row_number())
df1 |> right_join(df2)
df1 |> inner_join(df2)
df1 |> full_join(df2)
df1 |> semi_join(df2)
df1 |> anti_join(df2)