"""
módulo: feature_engineering.py

Este módulo provee funciones para generación de características usadas
en los diferentes modelos.

Para facilitar la extensión a nuevas estrategias, se implementa el
patrón Abstract Factory.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from datetime import datetime
from workalendar.america import Mexico

# ────────────────────────────────────────────────────────────────────────────#
#                           1. PRODUCTO ABSTRACTO                             #
# ────────────────────────────────────────────────────────────────────────────#


class BaseEngine(ABC):
    """
    Interfaz abstracta para estrategias de generación de características.
    Todas las estrategias concretas deben heredar de esta clase y definir el
    método generate.
    """
    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera las características a utilizar en el modelo.
        """
        pass

# ────────────────────────────────────────────────────────────────────────────#
#                           2. PRODUCTOS CONCRETOS                            #
# ────────────────────────────────────────────────────────────────────────────#


class DatesEngine(BaseEngine):
    """
    Estrategia de generación de características basadas en la fecha.
    Parámetros:
      - date_column: nombre de la columna de fecha.
    """
    def __init__(self, *, date_column: str = 'FECHA'):
        self.date_column = date_column
        self.__calendar__ = Mexico()

    def generate(self, base: pd.DataFrame) -> pd.DataFrame:
        if self.date_column not in base.columns:
            raise ValueError(
                f'No se encontró la columna {self.date_column} '
                'en la base.'
            )
        df = base[[self.date_column]]
        df = df.rename(columns={self.date_column: 'fecha'})
        df = df[['fecha']]
        df['año'] = df['fecha'].dt.year
        df['mes'] = df['fecha'].dt.month
        df['día_mes'] = df['fecha'].dt.day
        df['día_semana'] = df['fecha'].dt.weekday
        df['día_año'] = df['fecha'].dt.dayofyear
        df['inicio_mes'] = (
            df['fecha']
            .apply(
                lambda d:
                datetime(d.year, d.month, 1).weekday()
            )
        )
        df['día_festivo'] = (
            df['fecha']
            .apply(
                lambda d:
                not self.__calendar__.is_working_day(d)
            )
        )
        df['quincena'] = (
            df['día_mes']
            .apply(lambda d: 1 if d <= 15 else 2)
        )
        df['buen_fin'] = (
            df['fecha']
            .apply(
                lambda d:
                d.month == 11 and 17 <= d.day <= 20
                and d.weekday() == 4
            )
        )
        df['semana_año'] = (
            df['fecha']
            .dt.isocalendar().week.astype(int)
        )
        feature_columns = [
            c for c in df.columns
            if c != 'fecha'
        ]
        return df[feature_columns]


class ShiftsEngine(BaseEngine):
    """
    Estrategia de generación de características basadas en shifts de features.
    Parámetros:
      - lags: lista de enteros para los lags.
      - date_column, service_column, feature_column: nombres de columnas.
    """
    def __init__(
            self,
            *,
            lags=[1, 2, 3, 7, 365],
            date_column='FECHA',
            feature_column='RECIBIDAS',
            service_column=None,
    ):
        self.lags = lags
        self.special_columns = {
            'date': date_column,
            'feature': feature_column
        }
        if service_column is not None:
            self.special_columns['servicio'] = service_column

    def generate(self, base: pd.DataFrame) -> pd.DataFrame:
        for col in self.special_columns.values():
            if col not in base.columns:
                raise ValueError(
                    f'No se encontró la columna {col} '
                    'en la base.'
                )

        df = base.copy()
        feature_column = self.special_columns['feature']
        date_column = self.special_columns['date']
        if 'servicio' not in self.special_columns:
            df = df.sort_values(by=date_column)
            for lag in self.lags:
                df[f'{feature_column}_lag_{lag}'] = (
                    df[feature_column]
                    .shift(lag)
                )
            df = df[[
                f'{feature_column}_lag_{lag}'
                for lag in self.lags
            ]]
            return df.sort_index()

        df = df.rename(
            columns={
                self.date_column: 'fecha',
                self.service_column: 'servicio',
                self.feature_column: 'feature'
            }
        )
        df = df[['fecha', 'servicio', 'feature']]
        grp = df.groupby('servicio', observed=False)['feature']
        for lag in self.lags:
            df[f'{self.feature_column}_shift_{lag}'] = grp.transform(
                lambda x: x.shift(lag)
            )
        feature_columns = [
            c for c in df.columns
            if c not in ['servicio', 'fecha', 'feature']
        ]
        return df[feature_columns]


class RollingMeansEngine(BaseEngine):
    """
    Estrategia de generación de características basadas en
    medias móviles de features.
    """
    def __init__(
            self,
            *,
            windows: list[int] = [3, 7, 15, 30],
            date_column: str = 'FECHA',
            feature_column: str = 'RECIBIDAS',
            service_column: str = None,
    ):
        self.windows = windows
        self.special_columns = {
            'date': date_column,
            'feature': feature_column
        }
        if service_column is not None:
            self.special_columns['servicio'] = service_column

    def generate(self, base: pd.DataFrame) -> pd.DataFrame:
        for col in self.special_columns.values():
            if col not in base.columns:
                raise ValueError(
                    f'No se encontró la columna {col} '
                    'en la base.'
                )

        df = base.copy()
        feature_column = self.special_columns['feature']
        date_column = self.special_columns['date']
        if 'servicio' not in self.special_columns:
            df = df.sort_values(by=date_column)
            for win in self.windows:
                df[f'{feature_column}_ma_{win}'] = (
                    df[feature_column]
                    .rolling(win)
                    .mean()
                )
            df = df[[
                f'{feature_column}_ma_{win}'
                for win in self.windows
            ]]
            return df.sort_index()

        df = df.rename(
            columns={
                self.date_column: 'fecha',
                self.service_column: 'servicio',
                self.feature_column: 'feature'
            }
        )
        df = df[['fecha', 'servicio', 'feature']]
        grp = df.groupby('servicio', observed=False)['feature']
        for win in self.windows:
            df[f'{self.feature_column}_ma_{win}'] = (
                grp
                .transform(lambda x: x.rolling(win).mean())
            )
        feature_columns = [
            c for c in df.columns
            if c not in ['servicio', 'fecha', 'feature']
        ]
        return df[feature_columns]


class RatiosEngine(BaseEngine):
    """
    Estrategia de generación de características basadas en ratios de features.
    """
    def __init__(
            self,
            *,
            periods: list[int] = [1, 7, 365],
            date_column: str = 'FECHA',
            feature_column: str = 'RECIBIDAS',
            service_column: str = None,
    ):
        self.periods = periods
        self.special_columns = {
            'date': date_column,
            'feature': feature_column
        }
        if service_column is not None:
            self.special_columns['servicio'] = service_column

    def generate(self, base: pd.DataFrame) -> pd.DataFrame:
        for col in self.special_columns.values():
            if col not in base.columns:
                raise ValueError(
                    f'No se encontró la columna {col} '
                    'en la base.'
                )

        df = base.copy()
        feature_column = self.special_columns['feature']
        date_column = self.special_columns['date']
        if 'servicio' not in self.special_columns:
            df = df.sort_values(by=date_column)
            for p in self.periods:
                previous = (
                    df[feature_column]
                    .shift(p)
                    .astype(float)
                )
                current = (
                    df[feature_column]
                    .astype(float)
                )
                df[f'{feature_column}_ratio_{p}'] = np.where(
                    previous.isna(),
                    np.nan,
                    np.where(
                        previous == 0,
                        current,
                        current / previous
                    )
                )
            df = df[[
                f'{feature_column}_ratio_{p}'
                for p in self.periods
            ]]
            return df.sort_index()

        df = df.rename(
            columns={
                self.date_column: 'fecha',
                self.service_column: 'servicio',
                self.feature_column: 'feature'
            }
        )
        df = df[['fecha', 'servicio', 'feature']]
        grp = df.groupby('servicio', observed=False)['feature']
        for p in self.periods:
            df[f'{self.feature_column}_ratio_{p}'] = (
                df['feature'] / grp.transform(lambda x: x.shift(p))
            )
        feature_columns = [
            c for c in df.columns
            if c not in ['servicio', 'fecha', 'feature']
        ]
        return df[feature_columns]


class ShiftedRollingMeansEngine(BaseEngine):
    """
    Estrategia de generación de características basadas en medias móviles
    de features con un shift de lag 1.
    """
    def __init__(
            self,
            *,
            windows: list[int] = [7, 30],
            date_column: str = 'FECHA',
            feature_column: str = 'RECIBIDAS',
            service_column: str = None,
    ):
        self.windows = windows
        self.special_columns = {
            'date': date_column,
            'feature': feature_column
        }
        if service_column is not None:
            self.special_columns['servicio'] = service_column

    def generate(self, base: pd.DataFrame) -> pd.DataFrame:
        for col in self.special_columns.values():
            if col not in base.columns:
                raise ValueError(
                    f'No se encontró la columna {col} '
                    'en la base.'
                )

        df = base.copy()
        feature_column = self.special_columns['feature']
        date_column = self.special_columns['date']
        if 'servicio' not in self.special_columns:
            df = df.sort_values(by=date_column)
            for win in self.windows:
                df[f'{feature_column}_shift_1_ma_{win}'] = (
                    df[feature_column]
                    .shift(1)
                    .rolling(win)
                    .mean()
                )
            df = df[[
                f'{feature_column}_shift_1_ma_{win}'
                for win in self.windows
            ]]
            return df.sort_index()

        df = df.rename(
            columns={
                self.date_column: 'fecha',
                self.service_column: 'servicio',
                self.feature_column: 'feature'
            }
        )
        df = df[['fecha', 'servicio', 'feature']]
        grp = df.groupby('servicio', observed=False)['feature']
        for win in self.windows:
            df[f'{self.feature_column}_shift_1_ma_{win}'] = (
                grp
                .transform(lambda x: x.shift(1).rolling(win).mean())
            )
        feature_columns = [
            c for c in df.columns
            if c not in ['servicio', 'fecha', 'feature']
        ]
        return df[feature_columns]

# ────────────────────────────────────────────────────────────────────────────#
#                           3. FÁBRICA ABSTRACTA                              #
# ────────────────────────────────────────────────────────────────────────────#


class EngineFactory(ABC):
    """
    Abstract Factory que devuelve instancias de diferentes estrategias
    de imputación basadas en una cadena que indica el método deseado.
    """
    @abstractmethod
    def create_engine(self, **kwargs) -> BaseEngine:
        pass

# ────────────────────────────────────────────────────────────────────────────#
#                           4. FÁBRICAS CONCRETAS                             #
# ────────────────────────────────────────────────────────────────────────────#


class DatesFactory(EngineFactory):
    def create_engine(self, **kwargs) -> DatesEngine:
        return DatesEngine(**kwargs)


class ShiftsFactory(EngineFactory):
    def create_engine(self, **kwargs) -> ShiftsEngine:
        return ShiftsEngine(**kwargs)


class RollingMeansFactory(EngineFactory):
    def create_engine(self, **kwargs) -> RollingMeansEngine:
        return RollingMeansEngine(**kwargs)


class RatiosFactory(EngineFactory):
    def create_engine(self, **kwargs) -> RatiosEngine:
        return RatiosEngine(**kwargs)


class ShiftedRollingMeansFactory(EngineFactory):
    def create_engine(self, **kwargs) -> ShiftedRollingMeansEngine:
        return ShiftedRollingMeansEngine(**kwargs)

# ────────────────────────────────────────────────────────────────────────────#
#                               5. CLIENTE                                    #
# ────────────────────────────────────────────────────────────────────────────#


class EngineeringClient:
    """
    Cliente que construye las características a partir de un mapa:
      feature_factory_map: dict[
        nombre -> (factory_instance, kwargs_para_engine)
      ]
    """
    def __init__(
            self,
            *,
            feature_factory_map: dict = dict()
    ):
        self.feature_factory_map = feature_factory_map

    def build_features(self, base: pd.DataFrame) -> pd.DataFrame:
        df = base.copy()
        feats = []
        for name, (factory, params) in self.feature_factory_map.items():
            engine = factory.create_engine(**params)
            feat_df = engine.generate(df)
            feats.append(feat_df)
        if feats:
            df = pd.concat([*feats], axis=1)
        return df