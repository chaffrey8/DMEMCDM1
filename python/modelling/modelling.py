"""
módulo: modelling.py

Este módulo provee funciones para generación de características usadas
en los diferentes modelos.

Para facilitar la extensión a nuevas estrategias, se implementa el
patrón Abstract Factory.
"""

from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from sklearn.model_selection import GridSearchCV

# ────────────────────────────────────────────────────────────────────────────#
#                           1. PRODUCTO ABSTRACTO                             #
# ────────────────────────────────────────────────────────────────────────────#


class BaseModel(ABC):
    """
    Interfaz abstracta para estrategias de modelado.
    Todas las estrategias concretas deben heredar de esta clase y definir fit
    y predict.
    """
    @abstractmethod
    def fit(self, X, y=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass

# ────────────────────────────────────────────────────────────────────────────#
#                           2. PRODUCTOS CONCRETOS                            #
# ────────────────────────────────────────────────────────────────────────────#


class _FittedMixin:
    def __init__(self):
        self._is_fitted = False

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Debe llamar a fit() antes de predict().")


class MLPModel(_FittedMixin, BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = MLPRegressor(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        self._is_fitted = True

    def predict(self, X):
        self._check_fitted()
        return self.model.predict(X)


class RandomForestModel(_FittedMixin, BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = RandomForestRegressor(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        self._is_fitted = True

    def predict(self, X):
        self._check_fitted()
        return self.model.predict(X)


class LinearRegressionModel(_FittedMixin, BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = LinearRegression(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        self._is_fitted = True

    def predict(self, X):
        self._check_fitted()
        return self.model.predict(X)


class SVRModel(_FittedMixin, BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = SVR(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        self._is_fitted = True

    def predict(self, X):
        self._check_fitted()
        return self.model.predict(X)


class XGBRegressorModel(_FittedMixin, BaseModel):
    def __init__(self, **kwargs):
        super().__init__()

        parameters = {
            "n_estimators": [100, 200, 500],
            "max_depth": [3, 5, 10, 15],
            "learning_rate": [0.01, 0.05, 0.1]
            }
        self.model = GridSearchCV(
            XGBRegressor(objective='count:poisson', **kwargs),
            parameters
        )
        # self.model = XGBRegressor(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        self._is_fitted = True

    def predict(self, X):
        self._check_fitted()
        preds = self.model.predict(X)
        int_preds = np.round(np.clip(preds, 0, None)).astype(int)
        return int_preds


class ProphetModel(_FittedMixin, BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = Prophet(**kwargs)

    def fit(self, df):
        # df debe incluir columnas 'ds' (fecha) y 'y' (valor)
        self.model.fit(df)
        self._is_fitted = True

    def predict(self, df):
        self._check_fitted()
        return self.model.predict(df)


class ARIMAModel(_FittedMixin, BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.__kwargs__ = kwargs
        self._fitted_model = None

    def fit(self, ts):
        model = ARIMA(ts, self.__kwargs__)
        self._fitted_model = model.fit()
        self._is_fitted = True

    def predict(self, steps):
        self._check_fitted()
        return self._fitted_model.forecast(steps)


class LSTMModel(_FittedMixin, BaseModel):
    def __init__(self, *, epochs=10, batch_size=16, **layer_kwargs):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.layer_kwargs = layer_kwargs
        self.model = None

    def fit(self, X, y):
        # reshape X a (samples, timesteps, features)
        X_r = X.reshape((X.shape[0], 1, X.shape[1]))
        self.model = Sequential()
        self.model.add(
            LSTM(
                units=50,
                **self.layer_kwargs,
                input_shape=(1, X.shape[1])
            )
        )
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(X_r, y, epochs=self.epochs, batch_size=self.batch_size)
        self._is_fitted = True

    def predict(self, X):
        self._check_fitted()
        X_r = X.reshape((X.shape[0], 1, X.shape[1]))
        return self.model.predict(X_r)


class EnsembleModel(_FittedMixin, BaseModel):
    def __init__(self, meta_model=None, *base_models):
        super().__init__()
        self.base_models = base_models
        self.meta_model = meta_model or LinearRegression()

    def fit(self, X, y):
        preds = []
        for m in self.base_models:
            if not m._is_fitted:
                m.fit(X, y)
            preds.append(m.predict(X))
        meta_X = np.column_stack(preds)
        self.meta_model.fit(meta_X, y)
        self._is_fitted = True

    def predict(self, X):
        self._check_fitted()
        preds = [m.predict(X) for m in self.base_models]
        meta_X = np.column_stack(preds)
        return self.meta_model.predict(meta_X)

# ────────────────────────────────────────────────────────────────────────────#
#                           3. FÁBRICA ABSTRACTA                              #
# ────────────────────────────────────────────────────────────────────────────#


class AbstractModelFactory(ABC):
    """
    Fábrica abstracta para crear modelos.
    """
    @abstractmethod
    def create_model(self, **kwargs) -> BaseModel:
        pass

# ────────────────────────────────────────────────────────────────────────────#
#                           4. FÁBRICAS CONCRETAS                             #
# ────────────────────────────────────────────────────────────────────────────#


class MLPFactory(AbstractModelFactory):
    def create_model(self, **kwargs) -> MLPModel:
        return MLPModel(**kwargs)


class RandomForestFactory(AbstractModelFactory):
    def create_model(self, **kwargs) -> RandomForestModel:
        return RandomForestModel(**kwargs)


class LinearRegressionFactory(AbstractModelFactory):
    def create_model(self, **kwargs) -> LinearRegressionModel:
        return LinearRegressionModel(**kwargs)


class SVRFactory(AbstractModelFactory):
    def create_model(self, **kwargs) -> SVRModel:
        return SVRModel(**kwargs)


class XGBRegressorFactory(AbstractModelFactory):
    def create_model(self, **kwargs) -> XGBRegressorModel:
        return XGBRegressorModel(**kwargs)


class ProphetFactory(AbstractModelFactory):
    def create_model(self, **kwargs) -> ProphetModel:
        return ProphetModel(**kwargs)


class ARIMAFactory(AbstractModelFactory):
    def create_model(self, **kwargs) -> ARIMAModel:
        return ARIMAModel(**kwargs)


class LSTMFactory(AbstractModelFactory):
    def create_model(self, **kwargs) -> LSTMModel:
        return LSTMModel(**kwargs)


class EnsembleFactory(AbstractModelFactory):
    def create_model(self, meta_model=None, *base_models) -> EnsembleModel:
        return EnsembleModel(meta_model, *base_models)

# ────────────────────────────────────────────────────────────────────────────#
#                               5. CLIENTE                                    #
# ────────────────────────────────────────────────────────────────────────────#


class ModelClient:
    """
    Cliente para instanciar, entrenar y predecir con modelos configurables,
    incluyendo soporte directo para stacking ensembles basados en conjuntos de
    features.

    Ejemplo uso básico:
        client = ModelClient('mlp', hidden_layer_sizes=(50,))
        client.fit(X_train, y_train)
        preds = client.predict(X_test)

    Ejemplo ensemble con feature_sets:
        ensemble_client = ModelClient.create_ensemble(
            meta_model=LinearRegression(),
            base_model_type='xgboost',
            feature_sets=[
                ['dia_semana','mes','dia_mes','semana_anio'],
                ['RECIBIDAS_lag_1','RECIBIDAS_lag_7','RECIBIDAS_lag_365'],
                ['ratio_interdiario','ratio_intersemanal','ratio_interanual'],
                ['rolling_7','rolling_30']
            ],
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        ensemble_client.fit(X_train, y_train)
        y_pred = ensemble_client.predict(X_test)
    """
    _factory_map = {
        'mlp': MLPFactory(),
        'randomforest': RandomForestFactory(),
        'linearregression': LinearRegressionFactory(),
        'svr': SVRFactory(),
        'xgboost': XGBRegressorFactory(),
        'prophet': ProphetFactory(),
        'arima': ARIMAFactory(),
        'lstm': LSTMFactory(),
    }

    def __init__(self, model_type: str, **model_kwargs):
        key = model_type.lower()
        if key not in self._factory_map:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")
        self.is_ensemble = False
        self.model = self._factory_map[key].create_model(**model_kwargs)
        self.model_type = key

    @classmethod
    def create_ensemble(
        self,
        meta_model,
        base_model_type: str,
        feature_sets: list,
        **base_model_kwargs
    ):
        """
        Crea un cliente para stacking ensemble.
        - meta_model: instancia de sklearn-like con fit/predict.
        - base_model_type: clave en _factory_map para el tipo de base.
        - feature_sets: lista de listas de nombres de columnas.
        - base_model_kwargs: parámetros para cada modelo base.
        """
        # Instanciamos el cliente vacío
        client = self.__new__(self)
        client.is_ensemble = True
        client.meta_model = meta_model
        client.feature_sets = feature_sets
        # Creamos instancias de modelos base sin entrenar
        client.base_models = []
        factory = self._factory_map.get(base_model_type.lower())
        if factory is None:
            raise ValueError(
                "Tipo de modelo base no soportado: "
                f"{base_model_type}")
        for _ in feature_sets:
            client.base_models.append(
                factory.create_model(**base_model_kwargs)
            )
        return client

    @classmethod
    def create_ensemble_from_models(cls, meta_model, bases: list):
        """
        Crea un cliente para stacking ensemble a partir de instancias de
        modelos y sus feature_sets.

        Args:
            meta_model: instancia sklearn-like con métodos fit y predict.
            bases: lista de tuplas (model, feature_set), donde:
                - model: instancia de BaseModel ya configurada.
                - feature_set: lista de nombres de columnas para X.

        Usage:
            from modelling import ModelClient, XGBRegressorModel, SVRModel

            bases = [
                (XGBRegressorModel(...), ['f1','f2']),
                (SVRModel(...), ['f3','f4']),
            ]
            client = ModelClient.create_ensemble_from_models(
                                    LinearRegression(), bases
                                )
            client.fit(X_train, y_train)
            preds = client.predict(X_test)
        """
        client = cls.__new__(cls)
        client.is_ensemble = True
        client.meta_model = meta_model
        # Separamos modelos y sus feature_sets
        client.base_models = [m for m, _ in bases]
        client.feature_sets = [fs for _, fs in bases]
        return client

    def fit(self, X, y=None):
        """Entrena el modelo o ensemble.
        Para ensemble, X debe ser DataFrame completo y y vector/Series.
        """
        if not self.is_ensemble:
            if self.model_type == 'arima':
                return self.model.fit(X)
            # Modelos individuales: esperan X, y
            return self.model.fit(X, y)

        # Stacking ensemble: entrenar bases sobre cada subset y
        #  luego meta-modelo
        preds = []
        for fs, base in zip(self.feature_sets, self.base_models):
            base.fit(X[fs], y)
            preds.append(base.predict(X[fs]))
        meta_X = np.column_stack(preds)
        self.meta_model.fit(meta_X, y)
        return self

    def predict(self, X):
        """Genera predicciones.
        Para ensemble, X es DataFrame completo.
        """
        if not self.is_ensemble:
            return self.model.predict(X)

        preds = []
        for fs, base in zip(self.feature_sets, self.base_models):
            preds.append(base.predict(X[fs]))
        meta_X = np.column_stack(preds)
        return self.meta_model.predict(meta_X)