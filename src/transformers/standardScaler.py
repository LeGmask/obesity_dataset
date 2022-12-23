import numpy as np
import pandas as pd

from src.base.exceptions import NotFittedError
from src.base.transformer import BaseTransformer


class StandardScaler(BaseTransformer):
    """
    This is class is used to scale the data to a standard normal distribution.
    See base/transformer.py for more information about the api.
    """

    def __init__(self) -> None:
        """
        Initializes the StandardScaler class.
        """
        self.mean: int | None = None
        self.std: int | None = None

    def fit(self, X: np.ndarray | pd.DataFrame) -> None:
        """
        Fits the StandardScaler to the data by calculating the mean and standard deviation.

        :rtype: None
        :param X: the data to fit the StandardScaler to
        """
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
        """
        Transforms the data by scaling it to a standard normal distribution.

        :rtype: np.ndarray | pd.DataFrame
        :raises NotFittedError: if the StandardScaler has not been fitted yet
        :param X: the data to transform
        :return: the transformed data
        """
        if (self.mean is None) or (self.std is None):
            raise NotFittedError("StandardScaler has not been fitted yet.")

        return (X - self.mean) / self.std
