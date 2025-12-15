"""
módulo: outlier_detection.py

Este módulo provee funciones para detección de datos atípicos usando
diferentes estrategias:
1. MAD.
2. Z-score.
3. IQR.

Para facilitar la extensión a nuevas estrategias, se implementa el
patrón Abstract Factory.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

# ────────────────────────────────────────────────────────────────────────────#
#                           1. PRODUCTO ABSTRACTO                             #
# ────────────────────────────────────────────────────────────────────────────#


class BaseDetector(ABC):
    """
    Interface (clase abstracta) para detectores de outliers.
    Expone los métodos fit, detect, transform y fit_transform.
    """

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        """
        Ajusta el detector sobre un DataFrame de entrenamiento.
        Debe guardar internamente las estadísticas necesarias por columna.
        """
        pass

    @abstractmethod
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica el detector (con las estadísticas que se guardaron en fit)
        sobre un DataFrame de prueba y devuelve un DataFrame booleano
        indicando True en posiciones que son outliers.
        """
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica el detector sobre un DataFrame y devuelve un DataFrame con
        valores nulos en posiciones que son outliers.
        """
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Conveniencia: hace fit() y luego transform() sobre el mismo DataFrame.
        """
        self.fit(df)
        return self.transform(df)


# ────────────────────────────────────────────────────────────────────────────#
#                           2. PRODUCTOS CONCRETOS                            #
# ────────────────────────────────────────────────────────────────────────────#

class MADDetector(BaseDetector):
    """
    Detector de outliers basado en MAD (Median Absolute Deviation).
    """

    def __init__(self, *, threshold: float = 3.5, remainder='passthrough'):
        self.threshold = threshold
        self.medians_: dict[str, float] = {}
        self.mad_vals_: dict[str, float] = {}
        self.remainder = remainder
        self._is_fitted = False

    def fit(self, df: pd.DataFrame) -> None:
        self.cols_ = df.select_dtypes(include=[np.number]).columns.tolist()
        self.medians_.clear()
        self.mad_vals_.clear()
        for col in self.cols_:
            x = df[col].astype(float)
            med = x.median(skipna=True)
            abs_dev = (x - med).abs()
            mad_val = abs_dev.median(skipna=True)
            self.medians_[col] = med
            self.mad_vals_[col] = mad_val
        self._is_fitted = True

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError(
                "MADOutlierDetector: llama a fit() "
                "antes de transform()."
            )
        apply_cols = [c for c in self.cols_ if c in df.columns]
        result = pd.DataFrame(False, index=df.index, columns=apply_cols)
        for col in apply_cols:
            x = df[col].astype(float)
            med = self.medians_[col]
            mad_val = self.mad_vals_[col]
            if mad_val == 0 or np.isnan(mad_val):
                result[col] = False
                continue
            modified_z = 0.6745 * (x - med) / mad_val
            modified_z = modified_z.abs()
            result[col] = modified_z > self.threshold
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = self.detect(df)
        result = df[self.cols_].mask(result)
        if self.remainder == 'passthrough':
            restantes = [
                col for col in df.columns
                if col not in self.cols_
            ]
            result = pd.concat([result, df[restantes]], axis=1)[df.columns]
        elif self.remainder == 'drop':
            pass  # no se agregan
        else:
            raise ValueError("remainder debe ser 'passthrough' o 'drop'")
        return result


class ZScoreDetector(BaseDetector):
    """
    Detector de outliers basado en Z-score (puntuación estándar).
    """

    def __init__(self, *, threshold: float = 3.0, remainder='passthrough'):
        self.threshold = threshold
        self.means_: dict[str, float] = {}
        self.stds_: dict[str, float] = {}
        self.remainder = remainder
        self._is_fitted = False

    def fit(self, df: pd.DataFrame) -> None:
        self.cols_ = df.select_dtypes(include=[np.number]).columns.tolist()
        self.means_.clear()
        self.stds_.clear()
        for col in self.cols_:
            x = df[col].astype(float)
            mean = x.mean(skipna=True)
            std = x.std(skipna=True)
            self.means_[col] = mean
            self.stds_[col] = std
        self._is_fitted = True

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError(
                "ZScoreOutlierDetector: llama a fit() "
                "antes de transform()."
            )
        apply_cols = [c for c in self.cols_ if c in df.columns]
        result = pd.DataFrame(False, index=df.index, columns=apply_cols)
        for col in apply_cols:
            x = df[col].astype(float)
            mean = self.means_[col]
            std = self.stds_[col]
            if std == 0 or np.isnan(std):
                result[col] = False
                continue
            z_scores = ((x - mean) / std).abs()
            result[col] = z_scores > self.threshold
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = self.detect(df)
        result = df[self.cols_].mask(result)
        if self.remainder == 'passthrough':
            restantes = [
                col for col in df.columns
                if col not in self.cols_
            ]
            result = pd.concat([result, df[restantes]], axis=1)[df.columns]
        elif self.remainder == 'drop':
            pass  # no se agregan
        else:
            raise ValueError("remainder debe ser 'passthrough' o 'drop'")
        return result


class IQRDetector(BaseDetector):
    """
    Detector de outliers basado en IQR (Interquartile Range).
    """

    def __init__(self, *, threshold: float = 1.5, remainder='passthrough'):
        self.threshold = threshold
        self.q1_: dict[str, float] = {}
        self.q3_: dict[str, float] = {}
        self.iqr_vals_: dict[str, float] = {}
        self.remainder = remainder
        self._is_fitted = False

    def fit(self, df: pd.DataFrame, cols: list[str] | None = None) -> None:
        self.cols_ = df.select_dtypes(include=[np.number]).columns.tolist()
        self.q1_.clear()
        self.q3_.clear()
        self.iqr_vals_.clear()
        for col in self.cols_:
            x = df[col].astype(float)
            q1 = x.quantile(0.25)
            q3 = x.quantile(0.75)
            iqr_val = q3 - q1
            self.q1_[col] = q1
            self.q3_[col] = q3
            self.iqr_vals_[col] = iqr_val
        self._is_fitted = True

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError(
                "IQROutlierDetector: llama a fit() "
                "antes de transform()."
            )
        apply_cols = [c for c in self.cols_ if c in df.columns]
        result = pd.DataFrame(False, index=df.index, columns=apply_cols)
        for col in apply_cols:
            x = df[col].astype(float)
            q1 = self.q1_[col]
            q3 = self.q3_[col]
            iqr_val = self.iqr_vals_[col]
            if iqr_val == 0 or np.isnan(iqr_val):
                result[col] = False
                continue
            lower_bound = q1 - self.threshold * iqr_val
            upper_bound = q3 + self.threshold * iqr_val
            result[col] = (x < lower_bound) | (x > upper_bound)
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = self.detect(df)
        result = df[self.cols_].mask(result)
        if self.remainder == 'passthrough':
            restantes = [
                col for col in df.columns
                if col not in self.cols_
            ]
            result = pd.concat([result, df[restantes]], axis=1)[df.columns]
        elif self.remainder == 'drop':
            pass  # no se agregan
        else:
            raise ValueError("remainder debe ser 'passthrough' o 'drop'")
        return result

# ────────────────────────────────────────────────────────────────────────────#
#                           3. FÁBRICA ABSTRACTA                              #
# ────────────────────────────────────────────────────────────────────────────#


class AbstractOutlierFactory(ABC):
    """
    Fábrica abstracta para crear detectores de outliers.
    Define la firma create_detector() que devuelve un BaseOutlierDetector.
    """
    @abstractmethod
    def create_detector(self) -> BaseDetector:
        """
        Instancia y devuelve un detector concreto de outliers.
        """
        pass

# ────────────────────────────────────────────────────────────────────────────#
#                           4. FÁBRICAS CONCRETAS                             #
# ────────────────────────────────────────────────────────────────────────────#


class MADFactory(AbstractOutlierFactory):
    def create_detector(self) -> MADDetector:
        return MADDetector()


class ZScoreFactory(AbstractOutlierFactory):
    def create_detector(self) -> ZScoreDetector:
        return ZScoreDetector()


class IQROutlierFactory(AbstractOutlierFactory):
    def create_detector(self) -> IQRDetector:
        return IQRDetector()

# ────────────────────────────────────────────────────────────────────────────#
#                               5. CLIENTE                                    #
# ────────────────────────────────────────────────────────────────────────────#


class OutlierDetectionClient:
    def __init__(
            self,
            *,
            column_factory_map: dict = dict(),
            remainder='passthrough'
    ):
        """
        iterative: str con nombre de imputador iterativo
        column_factory_map: dict con nombre de columna -> factory (instancia)
        remainder: 'passthrough' o 'drop'
        """
        self.column_factory_map = column_factory_map
        self.remainder = remainder
        self.detectors = {}

    def fit(self, df: pd.DataFrame):
        for col, factory in self.column_factory_map.items():
            detector = factory.create_detector()
            detector.fit(df[[col]])
            self.detectors[col] = detector

    def detect(self, df: pd.DataFrame):
        result = pd.DataFrame()
        for col, detector in self.detectors.items():
            result[col] = detector.detect(df[[col]])
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame()
        for col, detector in self.detectors.items():
            result[col] = detector.transform(df[[col]])
        cols_aplicados = set(self.column_factory_map.keys())
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
        result = pd.DataFrame()
        # Imputar columnas específicas
        for col, detector in self.detectors.items():
            result[col] = (
                detector
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