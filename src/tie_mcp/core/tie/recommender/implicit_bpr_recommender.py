import numpy as np
import tensorflow as tf
from scipy import sparse
from sklearn.metrics import mean_squared_error

from tie.constants import PredictionMethod
from tie.utils import calculate_predicted_matrix

# Optional dependency: implicit (can be excluded in lightweight / CI environments)
try:  # pragma: no cover - import guard
    from implicit.bpr import BayesianPersonalizedRanking  # type: ignore
except Exception:  # pragma: no cover
    BayesianPersonalizedRanking = None  # type: ignore


class ImplicitBPRRecommender:
    """A matrix factorization recommender model to suggest items for an entity."""

    # Abstraction function:
    #   AF(m, n, k, model, num_new_users) = model to be trained with embedding dimension
    #   k
    #       on m entities and n items if model is None,
    #       or model trained on such data and with predictions for num_new_users
    #       if model is not None
    # Rep invariant:
    #   - m > 0
    #   - n > 0
    #   - k > 0
    #   - num_new_users >= 0
    # Safety from rep exposure:
    #   - k is private and immutable
    #   - model is never returned

    def __init__(self, m: int, n: int, k: int):
        """Initializes an ImplicitBPRRecommender object.

        Args:
            m: number of entities.  Requires m > 0.
            n: number of items.  Requires n > 0.
            k: the embedding dimension.  Requires k > 0.
        """
        # validate preconditions
        if m <= 0:
            raise ValueError(f"m must be > 0 (got {m})")
        if n <= 0:
            raise ValueError(f"n must be > 0 (got {n})")
        if k <= 0:
            raise ValueError(f"k must be > 0 (got {k})")

        self._m = m
        self._n = n
        self._k = k
        self._model = None

        self._num_new_users = 0

        self._checkrep()

    def _checkrep(self):
        """Validates the representation invariant; raises ValueError on violation."""
        if self._m <= 0:
            raise ValueError(f"Invalid state: m must be > 0 (got {self._m})")
        if self._n <= 0:
            raise ValueError(f"Invalid state: n must be > 0 (got {self._n})")
        if self._k <= 0:
            raise ValueError(f"Invalid state: k must be > 0 (got {self._k})")
        if self._num_new_users < 0:
            raise ValueError(
                f"Invalid state: num_new_users must be >= 0 (got {self._num_new_users})"
            )

    @property
    def U(self) -> np.ndarray:
        """Gets U as a factor of the factorization UV^T. Model must be trained."""
        if self._model is None:
            raise RuntimeError("Model must be trained before accessing user factors")
        self._checkrep()
        return np.copy(self._model.user_factors)

    @property
    def V(self) -> np.ndarray:
        """Gets V as a factor of the factorization UV^T. Model must be trained."""
        if self._model is None:
            raise RuntimeError("Model must be trained before accessing item factors")
        self._checkrep()
        return np.copy(self._model.item_factors)

    def fit(
        self,
        data: tf.SparseTensor,
        learning_rate: float,
        epochs: int,
        regularization_coefficient: float,
        **kwargs,
    ):
        """Fits the model to data.

        Args:
            data: An mxn tensor of training data.
            learning_rate: The learning rate.
                Requires learning_rate > 0.
            epochs: Number of training epochs, where each the model is trained on the
                cardinality dataset in each epoch.
            regularization_coefficient: Coefficient on the embedding regularization
                term.

        Mutates:
            The recommender to the new trained state.
        """

        if BayesianPersonalizedRanking is None:  # pragma: no cover - defensive
            raise ImportError(
                "implicit library not installed; install 'implicit' to use ImplicitBPRRecommender"
            )
        self._model = BayesianPersonalizedRanking(
            factors=self._k,
            learning_rate=learning_rate,
            regularization=regularization_coefficient,
            iterations=epochs,
            verify_negative_samples=True,
        )

        row_indices = tuple(index[0] for index in data.indices)
        column_indices = tuple(index[1] for index in data.indices)
        sparse_data = sparse.csr_matrix(
            (data.values, (row_indices, column_indices)), shape=data.shape
        )

        self._model.fit(sparse_data)

        self._checkrep()

    def evaluate(
        self,
        test_data: tf.SparseTensor,
        method: PredictionMethod = PredictionMethod.DOT,
    ) -> float:
        """Evaluates the solution.

        Requires that the model has been trained.

        Args:
            test_data: mxn tensor on which to evaluate the model.
                Requires that mxn match the dimensions of the training tensor and
                each row i and column j correspond to the same entity and item
                in the training tensor, respectively.
            method: The prediction method to use.

        Returns:
            The mean squared error of the test data.
        """
        predictions_matrix = self.predict(method)

        row_indices = tuple(index[0] for index in test_data.indices)
        column_indices = tuple(index[1] for index in test_data.indices)
        prediction_values = predictions_matrix[row_indices, column_indices]

        self._checkrep()
        return mean_squared_error(test_data.values, prediction_values)

    def predict(self, method: PredictionMethod = PredictionMethod.DOT) -> np.ndarray:
        """Gets the model predictions.

        The predictions consist of the estimated matrix A_hat of the truth
        matrix A, of which the training data contains a sparse subset of the entries.

        Args:
            method: The prediction method to use.

        Returns:
            An mxn array of values.
        """
        if self._model is None:
            raise RuntimeError("Model must be trained before calling predict")
        self._checkrep()
        return calculate_predicted_matrix(
            self._model.user_factors, self._model.item_factors, method
        )

    def predict_new_entity(
        self,
        entity: tf.SparseTensor,
        method: PredictionMethod = PredictionMethod.DOT,
        **kwargs,
    ) -> np.array:
        raise NotImplementedError
