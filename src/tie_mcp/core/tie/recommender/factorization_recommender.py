# Code adapted from https://colab.research.google.com/github/google/eng-edu/blob/main/ml/recommendation-systems/recommendation-systems.ipynb?utm_source=ss-recommendation-systems&utm_campaign=colab-external&utm_medium=referral&utm_content=recommendation-systems

import copy
from typing import Any

import numpy as np
from sklearn.metrics import mean_squared_error

# Optional heavy deps: keras / tensorflow. Guard import so package import
# does not fail in lightweight environments (e.g. CI without TF).
try:  # pragma: no cover - optional dependency
    import keras  # type: ignore
    import tensorflow as tf  # type: ignore

    _TF_AVAILABLE = True
    # Optional eager config (TF2 already eager). Ignore if API absent.
    try:  # pragma: no cover
        tf.config.run_functions_eagerly(True)
    except (AttributeError, RuntimeError) as _e:
        _ = _e  # Non-critical: API missing or already eager.
except Exception:  # pragma: no cover
    tf = None  # type: ignore
    keras = None  # type: ignore
    _TF_AVAILABLE = False

from ..constants import PredictionMethod
from ..utils import calculate_predicted_matrix
from .recommender import Recommender


class FactorizationRecommender(Recommender):
    """A matrix factorization collaborative filtering recommender model.

    TensorFlow / Keras optional:
      - Importing tie.recommender.* will not require TF.
      - Instantiating / training FactorizationRecommender requires TF/Keras.
    """

    # Abstraction function:
    #   AF(m, n, k) = a matrix factorization recommender model
    #       on m entities, n items to recommend, and
    #       embedding dimension k (a hyperparameter)
    # Rep invariant:
    #   - U.shape[1] == V.shape[1]
    #   - U and V are 2D
    #   - U.shape[0] > 0
    #   - U.shape[1] > 0
    #   - V.shape[0] > 0
    #   - V.shape[1] > 0
    #   - all elements of U are non-null
    #   - all elements of V are non-null
    #   - loss is not None
    # Safety from rep exposure:
    #   - U and V are private and not reassigned
    #   - methods to get U and V return a deepcopy of the numpy representation

    def __init__(self, m, n, k):
        """Initializes a FactorizationRecommender object.

        Args:
            m: number of entities
            n: number of items
            k: embedding dimension
        """
        if m <= 0:
            raise ValueError(f"m must be > 0 (got {m})")
        if n <= 0:
            raise ValueError(f"n must be > 0 (got {n})")
        if k <= 0:
            raise ValueError(f"k must be > 0 (got {k})")

        if not _TF_AVAILABLE:
            raise ImportError(
                "TensorFlow / Keras not installed; install tensorflow / keras to use FactorizationRecommender"
            )

        self._U = tf.Variable(tf.zeros((m, k)))  # type: ignore[attr-defined]
        self._V = tf.Variable(tf.zeros((n, k)))  # type: ignore[attr-defined]

        self._reset_embeddings()

        # keras available only if _TF_AVAILABLE
        self._loss = keras.losses.MeanSquaredError()  # type: ignore[union-attr]

        self._init_stddev = 1

        self._checkrep()

    def _require_tf(self):
        if not _TF_AVAILABLE:
            raise ImportError("TensorFlow not available; FactorizationRecommender operation requires it")

    def _reset_embeddings(self):
        """Resets the embeddings to a standard normal."""
        self._require_tf()
        init_stddev = 1

        new_U = tf.Variable(tf.random.normal(self._U.shape, stddev=init_stddev))  # type: ignore[attr-defined]
        new_V = tf.Variable(tf.random.normal(self._V.shape, stddev=init_stddev))  # type: ignore[attr-defined]

        self._U = new_U
        self._V = new_V

    def _checkrep(self):
        """Validates the rep invariant; raises ValueError on violation."""
        if not _TF_AVAILABLE:
            # If TF not available, object should never have been constructed.
            raise ValueError("Invalid state: TensorFlow unavailable for factorization model")
        if self._U.shape[1] != self._V.shape[1]:
            raise ValueError(
                f"Embedding dimension mismatch: U.shape[1]={self._U.shape[1]} V.shape[1]={self._V.shape[1]}"
            )
        if len(self._U.shape) != 2 or len(self._V.shape) != 2:
            raise ValueError(f"Embeddings must be 2D (got U.ndim={len(self._U.shape)}, V.ndim={len(self._V.shape)})")
        if self._U.shape[0] <= 0 or self._V.shape[0] <= 0:
            raise ValueError(
                f"Embedding counts must be > 0 (U.shape[0]={self._U.shape[0]}, V.shape[0]={self._V.shape[0]})"
            )
        if self._U.shape[1] <= 0 or self._V.shape[1] <= 0:
            raise ValueError(
                f"Embedding dim must be > 0 (U.shape[1]={self._U.shape[1]}, V.shape[1]={self._V.shape[1]})"
            )
        if tf.math.reduce_any(tf.math.is_nan(self._U)):  # type: ignore[attr-defined]
            raise ValueError("User embedding matrix U contains NaN values")
        if tf.math.reduce_any(tf.math.is_nan(self._V)):  # type: ignore[attr-defined]
            raise ValueError("Item embedding matrix V contains NaN values")
        if self._loss is None:
            raise ValueError("Loss function is not initialized")

    @property
    def U(self) -> np.ndarray:
        """Gets U as a factor of the factorization UV^T."""
        self._checkrep()
        return copy.deepcopy(self._U.numpy())  # type: ignore[union-attr]

    @property
    def V(self) -> np.ndarray:
        """Gets V as a factor of the factorization UV^T."""
        self._checkrep()
        return copy.deepcopy(self._V.numpy())  # type: ignore[union-attr]

    def _get_estimated_matrix(self) -> Any:
        """Gets the estimated matrix UV^T."""
        self._checkrep()
        return tf.matmul(self._U, self._V, transpose_b=True)  # type: ignore[attr-defined]

    def _predict(self, data: Any) -> Any:
        """Predicts the results for data.

        Requires that data be the same shape as the training data.
        Where each row corresponds to the same entity as the training data
        and each column represents the same item to recommend. However,
        the tensor may be sparse and contain more, fewer, or the same number
        of entries as the training data.

        Args:
            data: An mxn sparse-like tensor with .indices of nonzero entries.

        Returns:
            A length-p tensor of predictions corresponding to data.indices.
        """
        self._checkrep()
        return tf.gather_nd(self._get_estimated_matrix(), data.indices)  # type: ignore[attr-defined]

    def _calculate_regularized_loss(
        self,
        data: Any,
        predictions: Any,
        regularization_coefficient: float,
        gravity_coefficient: float,
    ) -> float:
        r"""Gets the regularized loss function.

        The regularized loss is the sum of:
        - The MSE between data and predictions.
        - A regularization term which is the average of the squared norm of each
            entity embedding, plus the average of the squared norm of each item
            embedding r = 1/m \sum_i ||U_i||^2 + 1/n \sum_j ||V_j||^2
        - A gravity term which is the average of the squares of all predictions.
            g = 1/(MN) \sum_{ij} (UV^T)_{ij}^2
        """
        regularization_loss = regularization_coefficient * (
            tf.reduce_sum(self._U * self._U) / self._U.shape[0]  # type: ignore[attr-defined]
            + tf.reduce_sum(self._V * self._V) / self._V.shape[0]  # type: ignore[attr-defined]
        )

        gravity = (1.0 / (self._U.shape[0] * self._V.shape[0])) * tf.reduce_sum(  # type: ignore[attr-defined]
            tf.square(tf.matmul(self._U, self._V, transpose_b=True))  # type: ignore[attr-defined]
        )

        gravity_loss = gravity_coefficient * gravity

        self._checkrep()
        return float(self._loss(data, predictions) + regularization_loss + gravity_loss)  # type: ignore[operator]

    def _calculate_mean_square_error(self, data: Any) -> Any:
        """Calculates the mean squared error between observed values in data and predictions from UV^T."""
        predictions = self._predict(data)
        loss = self._loss(data.values, predictions)  # type: ignore[union-attr]
        self._checkrep()
        return loss

    def fit(
        self,
        data: Any,
        learning_rate: float,
        epochs: int,
        regularization_coefficient: float = 0.1,
        gravity_coefficient: float = 0.0,
    ):
        """Fits the model to data.

        Args:
            data: an mxn sparse tensor (TensorFlow SparseTensor) of training data.
            epochs: Number of training epochs.
            learning_rate: the learning rate.
            regularization_coefficient: coefficient on embedding regularization term.
            gravity_coefficient: coefficient on prediction regularization term.
        """
        if not _TF_AVAILABLE:
            raise ImportError("TensorFlow not available; cannot train FactorizationRecommender")

        self._reset_embeddings()

        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)  # type: ignore[union-attr]

        for _i in range(epochs + 1):
            with tf.GradientTape():  # type: ignore[attr-defined]
                predictions = self._predict(data)
                loss = self._calculate_regularized_loss(
                    data.values,  # type: ignore[union-attr]
                    predictions,
                    regularization_coefficient,
                    gravity_coefficient,
                )
            gradients = tf.gradients(loss, [self._U, self._V])  # type: ignore[attr-defined]
            # Fallback if tf.gradients not available (eager), use GradientTape (already used)
            if gradients is None:  # pragma: no cover
                with tf.GradientTape() as tape2:  # type: ignore[attr-defined]
                    predictions2 = self._predict(data)
                    loss2 = self._calculate_regularized_loss(
                        data.values,  # type: ignore[union-attr]
                        predictions2,
                        regularization_coefficient,
                        gravity_coefficient,
                    )
                gradients = tape2.gradient(loss2, [self._U, self._V])  # type: ignore[attr-defined]
            optimizer.apply_gradients(zip(gradients, [self._U, self._V], strict=False))  # type: ignore[union-attr]

        self._checkrep()

    def evaluate(
        self,
        test_data: Any,
        method: PredictionMethod = PredictionMethod.DOT,
    ) -> float:
        """Evaluates the solution.

        Requires that the model has been trained.
        """
        if not _TF_AVAILABLE:
            raise ImportError("TensorFlow not available; cannot evaluate FactorizationRecommender")

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
        """
        if not _TF_AVAILABLE:
            raise ImportError("TensorFlow not available; cannot predict with FactorizationRecommender")
        self._checkrep()

        return calculate_predicted_matrix(
            np.nan_to_num(self._U.numpy()),
            np.nan_to_num(self._V.numpy()),
            method,  # type: ignore[union-attr]
        )

    def predict_new_entity(
        self,
        entity: Any,
        learning_rate: float,
        epochs: int,
        regularization_coefficient: float,
        gravity_coefficient: float,
        method: PredictionMethod = PredictionMethod.DOT,
    ) -> np.array:
        """Recommends items to an unseen entity."""
        if not _TF_AVAILABLE:
            raise ImportError("TensorFlow not available; cannot perform cold start prediction")
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)  # type: ignore[union-attr]

        embedding = tf.Variable(  # type: ignore[attr-defined]
            tf.random.normal(  # type: ignore[attr-defined]
                [self._U.shape[1], 1],
                stddev=self._init_stddev,
            )
        )

        for _i in range(epochs + 1):
            with tf.GradientTape() as tape:  # type: ignore[attr-defined]
                predictions = tf.matmul(self._V, embedding)  # type: ignore[attr-defined]
                loss = (
                    self._loss(
                        entity.values,
                        tf.gather_nd(predictions, entity.indices),  # type: ignore[union-attr]
                    )
                    + (
                        regularization_coefficient
                        * tf.reduce_sum(tf.math.square(embedding))  # type: ignore[attr-defined]
                        / self._U.shape[0]
                    )
                    + (
                        (gravity_coefficient / (self._U.shape[0] * self._V.shape[0]))
                        * tf.reduce_sum(tf.square(tf.matmul(self._V, embedding)))  # type: ignore[attr-defined]
                    )
                )
            gradients = tape.gradient(loss, [embedding])  # type: ignore[attr-defined]
            optimizer.apply_gradients(zip(gradients, [embedding], strict=False))  # type: ignore[union-attr]

        if np.isnan(embedding.numpy()).any():  # type: ignore[union-attr]
            raise ValueError("Embedding contains NaN values after optimization")
        self._checkrep()
        return np.squeeze(
            calculate_predicted_matrix(embedding.numpy().T, self._V.numpy(), method)  # type: ignore[union-attr]
        )


Recommender.register(FactorizationRecommender)
