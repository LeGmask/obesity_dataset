from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseTransformer(ABC):
	"""
	You will find 2 main methods:
	- fit: fits the transformer to the data
	- transform: transforms the data

	These methods are abstract and must be implemented by the child class.

	A third method is also available:
	- fit_transform: fits the transformer to the data and then transforms it, this is the most common use case and simply
		calls the fit and transform methods.
	"""

	@abstractmethod
	def fit(self, X: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
		"""
		Fits the transformer to the data.

		:rtype: np.ndarray | pd.DataFrame
		:param X: the data to fit the transformer to
		"""
		raise NotImplementedError

	@abstractmethod
	def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
		"""
		Transforms the data using the transformer.

		:param X: the data to transform
		:return: the transformed data
		"""
		raise NotImplementedError

	def fit_transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
		"""
		Fits the transformer to the data and then transforms it.

		:param X: the data to fit the transformer to and then transform
		:return: the transformed data
		"""
		self.fit(X)
		return self.transform(X)
