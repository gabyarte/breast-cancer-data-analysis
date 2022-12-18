import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


class AssignTransformer(BaseEstimator, TransformerMixin):
    """
    Class for applying `DataFrame.assign`, using a dict as
    unique parameter
    """
    def __init__(self, assign_map, copy=True):
        self.assign_map = assign_map
        self.copy = copy

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy() if self.copy else X
        return X_.assign(**self.assign_map)


class SimpleImputerTransformer(SimpleImputer):
    def __init__(self, missing_values=np.nan, strategy="mean", fill_value=None,
                 verbose="deprecated", copy=True, add_indicator=False):
        self.feature_names = None
        super().__init__(
            missing_values=missing_values,
            strategy=strategy,
            fill_value=fill_value,
            verbose=verbose,
            copy=copy,
            add_indicator=add_indicator)

    def fit( self, X, y=None ):
        X_ = X.copy()
        self.feature_names = list(X_.columns)
        return super().fit(X_)
    
    def transform(self, X, y=None):
        X_ = X.copy()
        _transform = pd.DataFrame(super().transform(X_), index=X_.index)
        _transform.columns = self.feature_names
        return _transform


class PandasColumnTransformer(ColumnTransformer):
    """
    Same functionality as sklearn's ColumnTransformer, but returns a DataFrame
    as an output.
    """
    def _hstack(self, Xs):
        if all(isinstance(x, pd.DataFrame) for x in Xs):
            return pd.concat(Xs, axis='columns', copy=False)
        else:
            return super()._hstack(Xs)
