from typing import List

import numpy as np
import pandas as pd
import scipy

from src.base.classifier import BaseClassifier


class ExpectationMaximisation(BaseClassifier):
    def __init__(self, k_clusters: int, max_iter: int = 100, tol: float = 1e-3):
        self.ml: List[float] = []
        self.labels_: List | None = None
        self.responsibilities = None
        self.parameters: None | List | np.ndarray = None
        self.k_clusters: int = k_clusters
        self.max_iter: int = max_iter
        self.tol: float = tol

    @staticmethod
    def get_random_psd_matrix(n: int):
        """
        Generate a random positive semi-definite matrix

        :param n: size of the matrix
        :return: random positive semi-definite matrix
        """
        random_matrix = np.random.randn(n, n)
        return np.dot(random_matrix.T, random_matrix)

    def init_random_parameters(self, data_shape: tuple):
        """
        Initialize the parameters of the model according to the data shape
        [[mu, sigma, pi] * k_clusters]

        :param data_shape: shape of the data
        :return: None
        """
        self.parameters = [
            [
                np.random.normal(0, 1, size=(data_shape[1],)),
                ExpectationMaximisation.get_random_psd_matrix(data_shape[1]),
                1 / self.k_clusters,
            ]
            for _ in range(self.k_clusters)
        ]

    @staticmethod
    def compute_gaussian(x, mu, sigma):
        """
        Compute the gaussian distribution

        :param x: data point
        :param mu: mean
        :param sigma: covariance matrix
        :return: probability
        """
        multivariate_normal = scipy.stats.multivariate_normal(mu, sigma)
        return multivariate_normal.pdf(x)

    def compute_responsibilities(self, X):
        self.responsibilities = np.zeros((X.shape[0], self.k_clusters))

        for i in range(X.shape[0]):
            for k in range(self.k_clusters):
                self.responsibilities[i, k] = self.parameters[k][
                    2
                ] * self.compute_gaussian(
                    X[i], self.parameters[k][0], self.parameters[k][1]
                )
            self.responsibilities[i, :] /= np.sum(self.responsibilities[i, :])

    def parameters_iter(self, X):
        for k in range(self.k_clusters):
            N_k = np.sum(self.responsibilities[:, k])
            self.parameters[k][0] = (1 / N_k) * np.sum(
                self.responsibilities[:, k].reshape(-1, 1) * X, axis=0
            )
            # print(np.outer(X - self.parameters[k][0], X - self.parameters[k][0]).shape)

            self.parameters[k][1] = 0
            for i in range(X.shape[0]):
                self.parameters[k][1] += self.responsibilities[i, k] * np.outer(
                    X[i] - self.parameters[k][0], X[i] - self.parameters[k][0]
                )

            self.parameters[k][1] /= N_k

            # self.parameters[k][1] = (1 / N_k) * np.sum(
            # 	self.responsibilities[:, k].reshape(-1, 1) * np.outer(X - self.parameters[k][0],
            # 														  X - self.parameters[k][0]),
            # 	axis=0)
            self.parameters[k][2] = N_k / X.shape[0]

    def fit(self, X: np.ndarray | pd.DataFrame) -> None:
        self.init_random_parameters(X.shape)
        self.ml = []

        try:
            for _ in range(self.max_iter):
                self.compute_responsibilities(X)
                self.ml.append(self.maximum_likelihood(X))
                self.parameters_iter(X)

                if len(self.ml) > 1 and np.abs(self.ml[-1] - self.ml[-2]) < self.tol:
                    print("Converged")
                    break

        except ValueError or np.linalg.LinAlgError:
            # sometimes the sum of the responsibilities is equal to 0 or
            # sometimes the covariance matrix is not positive semi-definite and the algorithm fails
            # so we just reinitialize the parameters and try again
            # @TODO: find a better way to handle this
            self.fit(X)

        self.predict(X)

    def predict(self, X):
        self.labels_ = np.argmax(self.responsibilities, axis=1)

    def maximum_likelihood(self, X):
        return np.sum(
            np.log(
                sum(
                    self.parameters[k][2]
                    * self.compute_gaussian(
                        X, self.parameters[k][0], self.parameters[k][1]
                    )
                    for k in range(self.k_clusters)
                )
            )
        )
