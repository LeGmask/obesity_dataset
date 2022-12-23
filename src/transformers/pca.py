import numpy as np
import pandas as pd

from src.base.exceptions import NotFittedError
from src.base.transformer import BaseTransformer


class PCA(BaseTransformer):
    def __init__(self, n_components: int) -> None:
        """
        Initializes the PCA with the number of components to use.
        The number of components must be smaller or equal to the number of features in the data.

        :param n_components:
        """
        self.n_components: int = n_components
        self.correlation_matrix: np.ndarray | None = None
        self.eigenvalues: np.ndarray | None = None
        self.eigenvectors: np.ndarray | None = None

    def fit(self, X: np.ndarray | pd.DataFrame) -> None:
        """
        Fits the PCA to the data by computing the correlation matrix and then computing the eigenvalues and eigenvectors
        of the correlation matrix.

        :param X: the data to fit the PCA to
        """

        if isinstance(X, pd.DataFrame):
            self.correlation_matrix = X.corr().to_numpy()
        else:
            self.correlation_matrix = 1 / len(X) * np.dot(X.T, X)

        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.correlation_matrix)

        # Sort the eigenvalues and eigenvectors in descending order by absolute value of the eigenvalues
        idx = self.eigenvalues.argsort()[::-1]
        self.eigenvalues = self.eigenvalues[idx]
        self.eigenvectors = self.eigenvectors[:, idx]

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
        """
        Transforms the data by projecting it onto the eigenvectors of the correlation matrix.
        Reduces the dimensionality of the data to the number of components specified in the constructor.

        :raises NotFittedError: if the PCA has not been fitted yet
        :param X: the data to transform
        :return: the transformed data
        """
        if self.correlation_matrix is None:
            raise NotFittedError("PCA has not been fitted yet.")

        return np.dot(X, self.eigenvectors[:, : self.n_components])

    def get_eigenvalues(self) -> np.ndarray:
        """
        Returns the eigenvalues of the correlation matrix used in the PCA.

        :return: returns a numpy array containing the eigenvalues of the PCA
        """
        if self.eigenvalues is None:
            raise NotFittedError("PCA has not been fitted yet.")

        return self.eigenvalues

    def set_n_components(self, n_components: int) -> None:
        """
        Sets the number of components to use in the PCA.

        :param n_components:
        """
        self.n_components = n_components

    def individual_quality(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if self.correlation_matrix is None:
            raise NotFittedError("PCA has not been fitted yet.")

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        alpha = np.dot(X, self.eigenvectors[:, 0])
        beta = np.dot(X, self.eigenvectors[:, 1])
        score = np.empty(len(X))

        for index, values in enumerate(X):
            score[index] = np.square(alpha[index]) / sum(np.square(values)) + np.square(
                beta[index]
            ) / sum(np.square(values))
        return score

    def get_saturations(self) -> np.ndarray:
        """
        Returns the saturations of the PCA.

        :return: returns a numpy array containing the saturations of the PCA
        """
        if self.eigenvalues is None:
            raise NotFittedError("PCA has not been fitted yet.")

        return np.array(
            [
                np.multiply(np.sqrt(self.eigenvalues), self.eigenvectors[j, :])
                for j in range(len(self.eigenvalues))
            ]
        )
