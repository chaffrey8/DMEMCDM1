import pandas as pd
from abc import ABC, abstractmethod

# Importamos las fábricas y clases base de los dos módulos que ya tienes:
from .outlier_detection import (
    MADFactory,
    ZScoreFactory,
    IQROutlierFactory,
    BaseOutlierDetector,
)
from .imputation import ImputerFactory, BaseImputer


# ────────────────────────────────────────────────────────────────────────────────
# 1. ABSTRACT FACTORY GENERAL PARA PREPROCESAMIENTO
# ────────────────────────────────────────────────────────────────────────────────

class AbstractPreprocessingFactory(ABC):
    """
    Fábrica abstracta que define cómo crear:
      1) Un detector de outliers (BaseOutlierDetector).
      2) Un imputer (BaseImputer).
    """

    @abstractmethod
    def create_outlier_detector(self) -> BaseOutlierDetector:
        """
        Debe devolver una instancia concreta de BaseOutlierDetector.
        """
        pass

    @abstractmethod
    def create_imputer(self) -> BaseImputer:
        """
        Debe devolver una instancia concreta de BaseImputer.
        """
        pass


class PreprocessingFactory(AbstractPreprocessingFactory):
    """
    Fábrica concreta que, según cadenas (strings) de 'outlier_strategy' e
    'imputer_strategy', arroja el detector e imputer correspondientes.

    outlier_strategy admite:    "mad", "zscore", "iqr".
    imputer_strategy admite:    "knn", "interpolation",
                                "bayesian_ridge", "random_forest".
    """

    def __init__(
        self,
        outlier_strategy: str = "iqr",
        imputer_strategy: str = "interpolation",
        outlier_kwargs: dict = None,
        imputer_kwargs: dict = None,
    ):
        self.outlier_strategy = outlier_strategy.lower()
        self.imputer_strategy = imputer_strategy.lower()
        self.outlier_kwargs = outlier_kwargs or {}
        self.imputer_kwargs = imputer_kwargs or {}

    def create_outlier_detector(self) -> BaseOutlierDetector:
        """
        Según el valor de self.outlier_strategy, retorna:
          - MADFactory(**outlier_kwargs).create_detector()
          - ZScoreFactory(**outlier_kwargs).create_detector()
          - IQROutlierFactory(**outlier_kwargs).create_detector()
        """
        if self.outlier_strategy == "mad":
            return MADFactory(**self.outlier_kwargs).create_detector()
        elif self.outlier_strategy == "zscore":
            return ZScoreFactory(**self.outlier_kwargs).create_detector()
        elif self.outlier_strategy == "iqr":
            return IQROutlierFactory(**self.outlier_kwargs).create_detector()
        else:
            raise ValueError(
                "Outlier strategy desconocida: "
                f"{self.outlier_strategy!r}"
            )

    def create_imputer(self) -> BaseImputer:
        """
        Utiliza la fábrica estática ImputerFactory para crear un imputador
        según self.imputer_strategy.
        """
        return ImputerFactory.get_imputer(
            self.imputer_strategy,
            **self.imputer_kwargs
        )


# ────────────────────────────────────────────────────────────────────────────────
# 2. CLASE PREPROCESSOR: ARMA Y EJECUTA EL PIPELINE
# ────────────────────────────────────────────────────────────────────────────────

class Preprocessor:
    """
    Toma un AbstractPreprocessingFactory, construye el detector de outliers
    y el imputador, y expone métodos para aplicar todo el pipeline de
    preprocesamiento (fit + transform) en conjuntos de datos.
    """

    def __init__(self, factory: AbstractPreprocessingFactory):
        # Creamos los objetos a partir de la fábrica:
        # type: BaseOutlierDetector
        self.outlier_detector = factory.create_outlier_detector()
        # type: BaseImputer
        self.imputer = factory.create_imputer()

    def fit(
        self,
        X_train: pd.DataFrame,
        *,
        outlier_cols: list[str] | None = None
    ) -> pd.DataFrame:
        """
        1) Ajusta el detector de outliers sobre X_train.
        2) Transforma X_train para marcar/eliminar outliers
           (según la lógica de cada detector).
        3) Ajusta el imputador sobre el X_train resultante
           (con outliers tratados).
        4) Devuelve X_train ya imputado.

        Parámetros:
          - X_train: DataFrame de entrenamiento.
          - outlier_cols: lista opcional de columnas numéricas específicas
                          sobre las que realizar detección de outliers.
                          Si es None, cada detector elegirá sus propias
                          columnas.

        Retorno:
          - DataFrame con outliers tratados y faltantes imputados.
        """
        # 1) Fit del detector de outliers
        #    Algunos detectores permiten pasar lista de columnas;
        #    si no, se ignora.
        try:
            # Intentamos pasar 'outlier_cols' (ZScore y MAD lo aceptan)
            self.outlier_detector.fit(X_train, cols=outlier_cols)
        except TypeError:
            # Si el detector no acepta 'cols', llamamos a fit solo con X_train
            self.outlier_detector.fit(X_train)

        # 2) Transform para marcar/tratar outliers en X_train
        X_train_no_outliers = self.outlier_detector.transform(X_train)

        # 3) Fit del imputador sobre el DataFrame post-outliers
        self.imputer.fit(X_train_no_outliers)

        # 4) Transform del imputador para obtener X_train imputado
        X_train_imputed = self.imputer.transform(X_train_no_outliers)
        return X_train_imputed

    def transform(
        self,
        X: pd.DataFrame,
        *,
        outlier_cols: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Aplica el pipeline de detección de outliers + transform del imputador
        sobre un nuevo DataFrame X (por ejemplo, el conjunto de test).

        1) Se asume que el detector ya fue .fit(...) sobre X_train.
        2) Se marcan/tratan outliers en X usando same logic que en train.
        3) Se imputan faltantes con el imputador ya ajustado.

        Parámetros:
          - X: DataFrame (test o validación).
          - outlier_cols: idem a fit().

        Retorno:
          - DataFrame con outliers tratados y faltantes imputados.
        """
        try:
            self.outlier_detector.fit(X, cols=outlier_cols)
            # (Nota: algunos detectores podrían querer retrain, pero en general
            #  aquí NO llamamos a fit de nuevo, sino solo a transform.
            #  Sin embargo, si insistes en pasar 'cols' y el detector lo pide
            #  en fit, convendría haber guardado 'cols' en fit del Preprocessor
            #  Ajusta según tu caso.)
        except TypeError:
            pass

        X_no_outliers = self.outlier_detector.transform(X)
        X_imputed = self.imputer.transform_test(X_no_outliers)
        return X_imputed

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        *,
        outlier_cols: list[str] | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Conveniencia para hacer todo en un paso:
          1) Ajustar detector + imputador sobre X_train.
          2) Devolver (X_train_preproc, X_test_preproc).

        X_train_preproc: se aplica fit(...) + transform(...) al train.
        X_test_preproc: se aplica transform(...) al test, usando los
                        detectores e imputadores ya ajustados.
        """
        # 1) Ajuste completo en X_train
        X_train_preprocessed = self.fit(X_train, outlier_cols=outlier_cols)

        # 2) Aplicamos la misma lógica al X_test
        X_test_preprocessed = self.transform(X_test, outlier_cols=outlier_cols)

        return X_train_preprocessed, X_test_preprocessed