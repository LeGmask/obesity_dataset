from abc import ABC, abstractmethod


class BaseClassifier(ABC):
	"""
	You will find 2 main methods:
	- fit: fits the classifier to the data
	- transform: predict the class of the data

	These methods are abstract and must be implemented by the child class.

	A third method is also available:
	- fit_predict: fits the transformer to the data and then predict it, this is the most common use case and simply
		calls the fit and predict methods.
	"""

	@abstractmethod
	def fit(self, X):
		raise NotImplementedError

	@abstractmethod
	def predict(self, X):
		raise NotImplementedError

	def fit_predict(self, X):
		self.fit(X)
		return self.predict(X)
