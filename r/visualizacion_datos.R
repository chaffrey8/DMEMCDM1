# Cargar tidyverse, que incluye ggplot2 y otras herramientas útiles
library(tidyverse)

# Gráfico de Dispersión (Scatter Plot)
mtcars %>%
  ggplot(aes(x = hp, y = mpg)) +
  geom_point() +
  labs(title = "Relación entre Caballos de Fuerza y Millas por Galón", 
       x = "Caballos de Fuerza (hp)", 
       y = "Millas por Galón (mpg)")

# Gráfico de dispersión con color y tamaño
mtcars %>%
  ggplot(aes(x = hp, y = mpg, color = factor(cyl), size = wt)) +
  geom_point() +
  labs(title = "Relación entre HP y MPG por Número de Cilindros", 
       x = "Caballos de Fuerza (hp)", 
       y = "Millas por Galón (mpg)")

# Gráfico de Línea (Line Plot)
economics %>%
  ggplot(aes(x = date, y = unemploy)) +
  geom_line(color = "blue") +
  labs(title = "Tasa de Desempleo en el Tiempo", 
       x = "Fecha", 
       y = "Desempleo")

# Gráfico de Línea con varias series
airquality %>%
  ggplot(aes(x = Day, y = Temp, color = factor(Month))) +
  geom_line() +
  labs(title = "Temperatura Diaria por Mes", 
       x = "Día", 
       y = "Temperatura")

# Histograma
mtcars %>%
  ggplot(aes(x = mpg)) +
  geom_histogram(binwidth = 2, fill = "skyblue", color = "black") +
  labs(title = "Distribución de Millas por Galón", 
       x = "Millas por Galón (mpg)", 
       y = "Frecuencia")

# Gráfico de Barras (Bar Plot)
mpg %>%
  ggplot(aes(x = class)) +
  geom_bar(fill = "orange") +
  labs(title = "Distribución de Tipos de Autos", 
       x = "Tipo de Auto", 
       y = "Frecuencia")

# Gráfico de Barras Apilado
mpg %>%
  ggplot(aes(x = class, fill = drv)) +
  geom_bar() +
  labs(title = "Distribución de Tipos de Autos por Tipo de Tracción", 
       x = "Tipo de Auto", 
       y = "Frecuencia")

# Boxplot
mpg %>%
  ggplot(aes(x = class, y = hwy)) +
  geom_boxplot(fill = "lightgreen") +
  labs(title = "Distribución del Rendimiento en Carretera por Tipo de Auto", 
       x = "Tipo de Auto", 
       y = "Rendimiento en Carretera (hwy)")

# Violin Plot
mpg %>%
  ggplot(aes(x = class, y = hwy)) +
  geom_violin(fill = "lightblue") +
  labs(title = "Distribución del Rendimiento en Carretera por Tipo de Auto", 
       x = "Tipo de Auto", 
       y = "Rendimiento en Carretera (hwy)")

# Gráfico de Densidad (Density Plot)
mpg %>%
  ggplot(aes(x = hwy)) +
  geom_density(fill = "purple", alpha = 0.5) +
  labs(title = "Distribución de Rendimiento en Carretera", 
       x = "Rendimiento en Carretera (hwy)", 
       y = "Densidad")

# Gráfico de Área (Area Plot)
economics %>%
  ggplot(aes(x = date, y = unemploy)) +
  geom_area(fill = "pink", color = "black") +
  labs(title = "Tasa de Desempleo en el Tiempo", 
       x = "Fecha", 
       y = "Desempleo")

# Gráficos de Facetas (Faceted Plot)
mpg %>%
  ggplot(aes(x = displ, y = hwy)) +
  geom_point() +
  facet_wrap(~ class) +
  labs(title = "Rendimiento en Carretera por Tamaño del Motor y Tipo de Auto", 
       x = "Desplazamiento del Motor", 
       y = "Rendimiento en Carretera (hwy)")

# Gráfico de Mapa de Calor (Heatmap)
# Crear matriz de correlación y reorganizar con tidyr
cor_data <- mtcars %>%
  cor() %>%
  as.data.frame() %>%
  rownames_to_column(var = "row") %>%
  pivot_longer(cols = -row, names_to = "column", values_to = "correlation")

# Gráfico de mapa de calor con ggplot2
cor_data %>%
  ggplot(aes(x = row, y = column, fill = correlation)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  labs(title = "Mapa de Calor de Correlación", x = "", y = "")
