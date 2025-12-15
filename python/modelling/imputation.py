"""
módulo: imputation.py

Este módulo provee funciones para imputación de datos faltantes usando
diferentes estrategias:
1. knn: K-Nearest Neighbors Imputer.
2. interpolation: imputación mediante interpolación de pandas.
3. bayesian_ridge: IterativeImputer con Bayesian Ridge.
4. random_forest: IterativeImputer con Random Forest.

Para facilitar la extensión a nuevas estrategias, se implementa el
patrón Abstract Factory.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import gc
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import KNNImputer as knn, IterativeImputer
from sklearn.linear_model import BayesianRidge
from typing import Optional

# ────────────────────────────────────────────────────────────────────────────#
#                           1. PRODUCTO ABSTRACTO                             #
# ────────────────────────────────────────────────────────────────────────────#


class BaseImputer(ABC):
    """
    Interfaz abstracta para estrategias de imputación.
    Todas las estrategias concretas deben heredar de esta clase y
    definir los métodos fit y transform.
    """

    @abstractmethod
    def fit(self, df: pd.DataFrame):
        """
        Ajusta la estrategia al conjunto de datos df (sin NaNs).
        """
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Imputa los valores faltantes en df y devuelve un nuevo DataFrame.
        """
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajusta la estrategia a df y luego imputa los valores faltantes.
        """
        self.fit(df)
        return self.transform(df)

# ────────────────────────────────────────────────────────────────────────────#
#                           2. PRODUCTOS CONCRETOS                            #
# ────────────────────────────────────────────────────────────────────────────#


class KNNImputer(BaseImputer):
    """
    Estrategia de imputación usando sklearn.impute.KNNImputer.
    """

    def __init__(self, *, n_neighbors: int = 5, weights: str = "uniform"):
        """
        Parámetros:
        - n_neighbors: número de vecinos a usar.
        - weights: 'uniform' o 'distance'.
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.imputer = knn(
            n_neighbors=self.n_neighbors,
            weights=self.weights
        )
        self.is_fitted = False

    def fit(self, X: pd.DataFrame):
        self.imputer.fit(X)
        self.is_fitted = True

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError(
                "El imputador debe ser entrenado con .fit() "
                "antes de usar .transform()"
            )
        imputed_array = self.imputer.transform(X)
        return pd.DataFrame(imputed_array, index=X.index, columns=X.columns)


class InterpolationImputer(BaseImputer):
    """
    Estrategia de imputación usando pandas.DataFrame.interpolate.
    No requiere ajuste (fit), ya que la interpolación se aplica directamente.
    """

    def __init__(
        self,
        *,
        method: str = "linear",
        axis: int = 0,
        limit_direction: str = "forward",
    ):
        """
        Parámetros:
        - method: 'linear', 'time', 'polynomial', etc.
        - axis: 0 para columnas, 1 para filas.
        - limit_direction: 'forward', 'backward' o 'both'.
        """
        self.method = method
        self.axis = axis
        self.limit_direction = limit_direction
        self.is_fitted = True

    def fit(self, df: pd.DataFrame):
        # No hace nada; la interpolación no requiere ajuste previo.
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.interpolate(
            method=self.method,
            axis=self.axis,
            limit_direction=self.limit_direction,
        )


class BayesianRidgeImputer(BaseImputer):
    """
    Estrategia de imputación usando IterativeImputer con
    estimator=BaysianRidge.
    """

    def __init__(self, *, max_iter: int = 10, tol: float = 1e-3):
        """
        Parámetros:
        - max_iter: número máximo de iteraciones para IterativeImputer.
        - tol: tolerancia de convergencia.
        """
        self.imputer = IterativeImputer(
            estimator=BayesianRidge(), max_iter=max_iter, tol=tol
        )
        self.is_fitted = False

    def fit(self, X: pd.DataFrame):
        self.imputer.fit(X)
        self.is_fitted = True

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError(
                "El imputador debe ser entrenado con .fit() "
                "antes de usar .transform()"
            )
        imputed_array = self.imputer.transform(X)
        return pd.DataFrame(imputed_array, index=X.index, columns=X.columns)


class RandomForestImputer(BaseImputer):
    """
    Estrategia de imputación usando IterativeImputer con
    estimator=RandomForestRegressor.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_depth: int = None,
        random_state: int = None,
        max_iter: int = 10,
        tol: float = 1e-3,
    ):
        """
        Parámetros:
        - n_estimators: número de árboles del RandomForestRegressor.
        - max_depth: profundidad máxima de los árboles.
        - random_state: semilla para garantizar reproducibilidad.
        - max_iter: número máximo de iteraciones para IterativeImputer.
        - tol: tolerancia de convergencia.
        """
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        self.imputer = IterativeImputer(
            estimator=rf,
            max_iter=max_iter,
            tol=tol
        )
        self.is_fitted = False

    def fit(self, X: pd.DataFrame):
        self.imputer.fit(X)
        self.is_fitted = True

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError(
                "El imputador debe ser entrenado con .fit() "
                "antes de usar .transform()"
            )
        imputed_array = self.imputer.transform(X)
        return pd.DataFrame(imputed_array, index=X.index, columns=X.columns)

# ────────────────────────────────────────────────────────────────────────────#
#                           3. FÁBRICA ABSTRACTA                              #
# ────────────────────────────────────────────────────────────────────────────#


class ImputerFactory(ABC):
    """
    Abstract Factory que devuelve instancias de diferentes estrategias
    de imputación basadas en una cadena que indica el método deseado.
    """

    @abstractmethod
    def create_imputer(self) -> BaseImputer:
        pass

# ────────────────────────────────────────────────────────────────────────────#
#                           4. FÁBRICAS CONCRETAS                             #
# ────────────────────────────────────────────────────────────────────────────#


class KNNFactory(ImputerFactory):
    def create_imputer(self, **kwargs) -> BaseImputer:
        return KNNImputer(**kwargs)


class InterpolationFactory(ImputerFactory):
    def create_imputer(self, **kwargs) -> BaseImputer:
        return InterpolationImputer(**kwargs)


class BayesianRidgeFactory(ImputerFactory):
    def create_imputer(self, **kwargs) -> BaseImputer:
        return BayesianRidgeImputer(**kwargs)


class RandomForestFactory(ImputerFactory):
    def create_imputer(self, **kwargs) -> BaseImputer:
        return RandomForestImputer(**kwargs)

# ────────────────────────────────────────────────────────────────────────────#
#                               5. CLIENTE                                    #
# ────────────────────────────────────────────────────────────────────────────#


class ImputationClient:
    def __init__(
            self,
            *,
            iterative: str = 'randomforest',
            column_factory_map: dict = dict(),
            remainder='passthrough',
            **kwargs
    ):
        """
        iterative: str con nombre de imputador iterativo
        column_factory_map: dict con nombre de columna -> factory (instancia)
        remainder: 'passthrough' o 'drop'
        """
        if iterative in ['randomforest', 'bayesianridge']:
            self.iterative = True
            self.column_factory_map = dict()
            self.remainder = remainder
            self.imputers = {
                'all': dict({
                    'randomforest': RandomForestFactory(**kwargs),
                    'bayesianridge': BayesianRidgeFactory(**kwargs)
                }).get(iterative)
            }
        else:
            self.iterative = False
            self.column_factory_map = column_factory_map
            self.remainder = remainder
            self.imputers = {}

    def fit(self, df: pd.DataFrame):
        if self.iterative:
            factory = self.imputers['all']
            imputer = factory.create_imputer()
            cols_aplicados = (
                df
                .select_dtypes(include=['number'])
                .columns
                .tolist()
            )
            imputer.fit(df[cols_aplicados])
            self.imputers['all'] = imputer
        else:
            for col, factory in self.column_factory_map.items():
                imputer = factory.create_imputer()
                imputer.fit(df[[col]])
                self.imputers[col] = imputer

    def transform(
            self,
            df: pd.DataFrame,
            chunk_size: Optional[int] = None,
            n_jobs: int = 1
    ) -> pd.DataFrame:
        """
        Imputa los valores faltantes en `df` de manera opcional en paralelo.

        :param df: DataFrame original.
        :param chunk_size: Tamaño de cada lote. Si None, divide en n_jobs trozos.
        :param n_jobs: Número de trabajos paralelos. Si 1, comportamiento secuencial.
        :return: DataFrame con datos imputados.
        """
        if self.iterative:
            imputer = self.imputers['all']
            cols_aplicados = (
                df
                .select_dtypes(include=['number'])
                .columns
                .tolist()
            )
            data = df[cols_aplicados]
            # Preparar batches
            if chunk_size is None and n_jobs > 1:
                # dividir en n_jobs partes aproximadamente iguales
                batches = np.array_split(data, n_jobs)
            elif chunk_size is not None:
                batches = [data.iloc[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
            else:
                batches = [data]

            # Transformación
            if n_jobs > 1:
                # Paralelizar usando joblib
                bloques = Parallel(n_jobs=n_jobs)(
                    delayed(imputer.transform)(batch) for batch in batches
                )
            else:
                # Secuencial
                bloques = [imputer.transform(batch) for batch in batches]
            # Concatenar resultados
            imputed = np.vstack(bloques)
            result = pd.DataFrame(imputed, index=df.index, columns=cols_aplicados)
        else:
            result = pd.DataFrame()
            # Imputar columnas específicas
            for col, imputer in self.imputers.items():
                result[col] = imputer.transform(df[[col]])
            cols_aplicados = set(self.column_factory_map.keys())
        # Manejar columnas restantes
        restantes = [
            col for col in df.columns
            if col not in cols_aplicados
        ]
        if self.remainder == 'passthrough':
            result = pd.concat([result, df[restantes]], axis=1)[df.columns]
        elif self.remainder == 'drop':
            pass  # no se agregan
        else:
            raise ValueError("remainder debe ser 'passthrough' o 'drop'")
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.iterative:
            imputer = self.imputers['all']
            cols_aplicados = (
                df
                .select_dtypes(include=['number'])
                .columns
                .tolist()
            )
            result = imputer.fit_transform(df[cols_aplicados])
        else:
            result = pd.DataFrame()
            # Imputar columnas específicas
            for col, imputer in self.imputers.items():
                result[col] = (
                    imputer
                    .fit_transform(df[[col]])
                )
            cols_aplicados = set(self.column_factory_map.keys())
        # Manejar columnas restantes
        restantes = [
            col for col in df.columns
            if col not in cols_aplicados
        ]
        if self.remainder == 'passthrough':
            result = pd.concat([result, df[restantes]], axis=1)[df.columns]
        elif self.remainder == 'drop':
            pass  # no se agregan
        else:
            raise ValueError("remainder debe ser 'passthrough' o 'drop'")
        return result