import numpy as np
import pandas as pd
from numpy import ndarray

from src.base.classifier import BaseClassifier


class KMeans(BaseClassifier):
    """
    This class implements the K-Means algorithm. This is a clustering algorithm that aims to partition n observations
    into k clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of
    the cluster.
    """

    def __init__(
        self,
        k: int = 2,
        tol: float = 0.0001,
        max_iter: int = 1000,
        centroids_init: str = "kmeans++",
    ) -> None:
        """
        Initializes the K-Means classifier.

        :param k: the number of clusters to use
        :param tol: the tolerance to use when checking for convergence
        :param max_iter: the maximum number of iterations to run
        :param centroids_init: the method to use when initializing the centroids
        """
        self.centroids_init = centroids_init
        self.classes = None
        self.centroids = None
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    @staticmethod
    def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the euclidean distance between two points.

        :param x: the first point
        :param y: the second point
        :return: The euclidean distance between x and y
        """
        return np.sqrt(np.sum(np.square(x - y), axis=0))

    def __centroids__iter(self, X: np.ndarray | pd.DataFrame) -> None:
        """
        Updates position of centroids. This is done by computing the mean of all points in a cluster and setting the
        centroid to that mean.

        :param X: the data to update the centroids with
        """
        self.__assign_labels(X)

        new_centroids = []

        for centroid in range(self.k):
            new_centroids.append(np.mean(X[np.array(self.labels_) == centroid], axis=0))

        self.centroids = np.array(new_centroids)

    def __init_centroids_randomly(self, x) -> None:
        """
        Generates random centroids from the data.

        :param x: the data to initialize the centroids with
        """
        self.centroids = np.array([x[np.random.randint(len(x))] for _ in range(self.k)])

    def __init_centroids_plusplus(self, X: np.ndarray | pd.DataFrame) -> None:
        """
        Initializes the centroids using the K-Means++ algorithm.

        :param X: the data to initialize the centroids with
        """
        self.centroids = [
            X[np.random.randint(len(X))]
        ]  # randomly select first centroid

        for _ in range(self.k - 1):
            distances = np.array(
                [np.min([np.sum((x - c) ** 2) for c in self.centroids]) for x in X]
            )
            probs = distances / distances.sum()

            self.centroids.append(X[np.random.choice(X.shape[0], p=probs.ravel())])

        self.centroids = np.array(self.centroids)

    def fit(self, X: np.ndarray | pd.DataFrame) -> None:
        """
        Fits the K-Means parameters to the data.
        This is done by computing centroids on X.

        :rtype: None
        :param X: the data to fit the K-Means to
        """
        match self.centroids_init:
            case "random":
                self.__init_centroids_randomly(X)
            case "kmeans++":
                self.__init_centroids_plusplus(X)

        for _ in range(self.max_iter):
            previous_centroids = np.copy(self.centroids)

            self.__centroids__iter(X)  # update centroids

            if np.all(np.isclose(self.centroids, previous_centroids, atol=self.tol)):
                print("Converged")
                break  # if we have an optimal solution according to the given tolerance, we can stop

        self.__assign_labels(X)  # assign labels

    def __assign_labels(self, X: np.ndarray | pd.DataFrame) -> None:
        """
        Assigns labels to the data. This is done by computing the euclidean distance between each point and each
        centroid and assigning the point to the centroid with the smallest distance.

        :param X: the data to assign labels to
        """
        self.labels_ = []
        for point in range(len(X)):
            idx = np.argmin(
                [
                    KMeans.euclidean_distance(X[point, :], centroid)
                    for centroid in self.centroids
                ]
            )

            self.labels_.append(idx)

    def predict(self, X: np.ndarray | pd.DataFrame) -> list[ndarray[int]]:
        """
        Predicts the class of the data

        :rtype: np.ndarray | pd.DataFrame
        :param X: the data to predict the class of
        :return: the predicted labels
        """
        self.__assign_labels(X)
        return self.labels_
