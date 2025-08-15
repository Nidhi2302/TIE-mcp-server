from typing import Any

import numpy as np
from sklearn.metrics import mean_squared_error

# Optional TF dependency (sparse tensor handling). Guard so module import
# does not fail when tensorflow is absent (e.g. slim CI environments).
try:  # pragma: no cover - optional dependency
    import tensorflow as tf  # type: ignore
    _TF_AVAILABLE = True
except Exception:  # pragma: no cover
    tf = None  # type: ignore
    _TF_AVAILABLE = False

from ..constants import PredictionMethod
from ..utils import calculate_predicted_matrix
from .recommender import Recommender


class WalsRecommender(Recommender):
    """A WALS matrix factorization collaborative filtering recommender model.

    TensorFlow optional:
      - Passing TensorFlow SparseTensor inputs requires tensorflow installed.
      - You may also pass a dense numpy ndarray of shape (m, n) to fit/evaluate.
    """

    # Abstraction function:
    # AF(U, V) = a matrix factorization collaborative filtering recommendation model
    #   with user embeddings U and item embeddings V
    # Rep invariant:
    #   - U is not None
    #   - V is not None
    # Safety from rep exposure:
    #   - k is private and immutable
    #   - model is never returned

    def __init__(self, m: int, n: int, k: int = 10):
        """Initializes a new WALSRecommender object.

        Args:
            m: number of entities.  Requires m > 0.
            n: number of items.  Requires n > 0.
            k: embedding dimension.  Requires k > 0.
        """
        if m <= 0:
            raise ValueError(f"m must be > 0 (got {m})")
        if n <= 0:
            raise ValueError(f"n must be > 0 (got {n})")
        if k <= 0:
            raise ValueError(f"k must be > 0 (got {k})")

        self._U = np.zeros((m, k))
        self._V = np.zeros((n, k))
        self._reset_embeddings()

        self._checkrep()

    def _reset_embeddings(self):
        """Resets the embeddings to a standard normal."""
        init_stddev = 1

        new_U = np.random.normal(loc=0, scale=init_stddev, size=self._U.shape)
        new_V = np.random.normal(loc=0, scale=init_stddev, size=self._V.shape)

        self._U = new_U
        self._V = new_V

    def _checkrep(self):
        """Validates the rep invariant; raises ValueError on violation."""
        if self._U is None:
            raise ValueError("Invalid state: _U must not be None")
        if self._V is None:
            raise ValueError("Invalid state: _V must not be None")

    @property
    def m(self) -> int:
        """Gets the number of entities represented by the model."""
        self._checkrep()
        return self._U.shape[0]

    @property
    def n(self) -> int:
        """Gets the number of items represented by the model."""
        self._checkrep()
        return self._V.shape[0]

    @property
    def k(self) -> int:
        """Gets the embedding dimension of the model."""
        self._checkrep()
        return self._U.shape[1]

    @property
    def U(self) -> np.ndarray:
        """Gets U as a factor of the factorization UV^T. Model must be trained."""
        self._checkrep()
        return np.copy(self._U)

    @property
    def V(self) -> np.ndarray:
        """Gets V as a factor of the factorization UV^T. Model must be trained."""
        self._checkrep()
        return np.copy(self._V)

    def _update_factor(
        self,
        opposing_factors: np.ndarray,
        data: np.ndarray,
        alpha: float,
        regularization_coefficient: float,
    ) -> np.ndarray:
        """Updates factors according to least squares on the opposing factors.

        Determines factors which minimize loss on data based on opposing_factors.
        For example, if opposing_factors are the item factors, determines the entity
        factors which minimize loss on data.

        Args:
            opposing_factors: a pxk array of the fixed factors in the optimization step
                (ie entity or item factors).  Requires p, k > 0.
            predictions: A pxq array of the observed values for each of the
                entities/items associated with the p opposing_factors and the q
                items/entities associated with factors. Requires p, q > 0.
            alpha: Weight for positive training examples such that each positive example
                takes value alpha + 1.  Requires alpha > 0.
            regularization_coefficient: coefficient on the embedding regularization
                term. Requires regularization_coefficient > 0.

        Returns:
            A qxk array of recomputed factors which minimize error.
        """
        # validate preconditions
        p, k = opposing_factors.shape
        q = data.shape[1]
        if p <= 0:
            raise ValueError(f"p must be > 0 (got {p})")
        if k != self.k:
            raise ValueError(f"Opposing factors k mismatch: expected {self.k}, got {k}")
        if p != data.shape[0]:
            raise ValueError(
                "Data row count mismatch: "
                f"opposing_factors p={p}, data.shape[0]={data.shape[0]}"
            )
        if q <= 0:
            raise ValueError(f"q must be > 0 (got {q})")
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0 (got {alpha})")
        if regularization_coefficient < 0:
            raise ValueError(
                "regularization_coefficient must be >= 0 "
                f"(got {regularization_coefficient})"
            )

        def V_T_C_I_V(V, c_array):
            _, k = V.shape

            c_minus_i = c_array - 1
            nonzero_c = tuple(np.nonzero(c_minus_i)[0].tolist())

            product = np.zeros((k, k))

            for i in nonzero_c:
                v_i = np.expand_dims(V[i, :], axis=1)

                square_addition = v_i @ v_i.T
                if square_addition.shape != (k, k):
                    raise RuntimeError(
                        "square_addition shape mismatch: "
                        f"expected {(k, k)}, got {square_addition.shape}"
                    )

                product += square_addition

            return product

        # in line with the paper,
        # we will use variable names as if we are updating user factors based
        # on V, the item factors.  Since the process is the same for both,
        # the variable names are interchangeable.  This just makes following
        # along with the paper easier.
        V = opposing_factors

        new_U = np.ndarray((q, k))
        # for each item embedding

        V_T_V = V.T @ V
        # update each of the q user factors
        for i in range(q):
            P_u = data[:, i]
            # C is c if unobserved, one otherwise
            C_u = np.where(P_u > 0, alpha + 1, 1)
            if C_u.shape != (p,):
                raise ValueError(
                    f"C_u shape mismatch: expected {(p,)}, got {C_u.shape}"
                )

            confidence_scaled_v_transpose_v = V_T_C_I_V(V, C_u)

            # X = (V^T CV + \lambda I)^{-1} V^T CP
            inv = np.linalg.inv(
                V_T_V
                + confidence_scaled_v_transpose_v
                + regularization_coefficient * np.identity(k)
            )

            # removed C_u here since unnecessary in binary case
            # P_u is already binary
            U_i = inv @ V.T @ P_u

            new_U[i, :] = U_i

        return new_U

    def _coerce_input_matrix(self, data: Any, expect_shape: tuple[int, int]) -> np.ndarray:
        """Convert supported input (TF sparse tensor or ndarray) to dense ndarray."""
        if hasattr(data, "indices") and hasattr(data, "values"):
            if not _TF_AVAILABLE:
                raise ImportError(
                    "TensorFlow required to pass a SparseTensor to WalsRecommender (not installed)"
                )
            dense = tf.sparse.to_dense(tf.sparse.reorder(data)).numpy()  # type: ignore[attr-defined]
        else:
            if not isinstance(data, np.ndarray):
                raise TypeError(
                    "data must be either a TensorFlow SparseTensor or a numpy ndarray"
                )
            dense = data
        if dense.shape != expect_shape:
            raise ValueError(
                f"Input matrix shape mismatch: expected {expect_shape}, got {dense.shape}"
            )
        return dense

    def fit(
        self,
        data: Any,
        epochs: int,
        c: float = 0.024,
        regularization_coefficient: float = 0.01,
    ):
        """Fits the model to data.

        Args:
            data: An mxn tensor of training data (TF SparseTensor or ndarray).
            epochs: Number of training epochs, where each the model is trained on the
                cardinality dataset in each epoch.
            c: Weight for negative training examples in the loss function,
                ie each positive example takes weight 1, while negative examples take
                discounted weight c.  Requires 0 < c < 1.
            regularization_coefficient: Coefficient on the embedding regularization
                term.

        Mutates:
            The recommender to the new trained state.
        """
        self._reset_embeddings()

        # preconditions
        if not (0 < c < 1):
            raise ValueError(f"c must satisfy 0 < c < 1 (got {c})")

        P: np.ndarray = self._coerce_input_matrix(data, (self.m, self.n))

        alpha = (1 / c) - 1

        for _ in range(epochs):
            # step 1: update U
            self._U = self._update_factor(
                self._V, P.T, alpha, regularization_coefficient
            )

            # step 2: update V
            self._V = self._update_factor(self._U, P, alpha, regularization_coefficient)

        self._checkrep()

    def evaluate(
        self,
        test_data: Any,
        method: PredictionMethod = PredictionMethod.DOT,
    ) -> float:
        """Evaluates the solution.

        Requires that the model has been trained.

        Args:
            test_data: mxn test data (TF SparseTensor or ndarray)
            method: The prediction method to use.

        Returns:
            The mean squared error of the test data.
        """
        predictions_matrix = self.predict(method)

        if hasattr(test_data, "indices") and hasattr(test_data, "values"):
            if not _TF_AVAILABLE:
                raise ImportError(
                    "TensorFlow required to evaluate with a SparseTensor (not installed)"
                )
            row_indices = tuple(index[0] for index in test_data.indices)
            column_indices = tuple(index[1] for index in test_data.indices)
            prediction_values = predictions_matrix[row_indices, column_indices]
            values = test_data.values
        else:
            if not isinstance(test_data, np.ndarray):
                raise TypeError(
                    "test_data must be a TensorFlow SparseTensor or a numpy ndarray"
                )
            if test_data.shape != (self.m, self.n):
                raise ValueError(
                    f"test_data shape mismatch: expected {(self.m, self.n)}, got {test_data.shape}"
                )
            prediction_values = predictions_matrix[test_data > 0]
            values = test_data[test_data > 0]

        self._checkrep()
        return mean_squared_error(values, prediction_values)

    def predict(self, method: PredictionMethod = PredictionMethod.DOT) -> np.ndarray:
        """Gets the model predictions.

        The predictions consist of the estimated matrix A_hat of the truth
        matrix A, of which the training data contains a sparse subset of the entries.

        Args:
            method: The prediction method to use.

        Returns:
            An mxn array of values.
        """
        self._checkrep()

        return calculate_predicted_matrix(self._U, self._V, method)

    def predict_new_entity(
        self,
        entity: Any,
        c: float,
        regularization_coefficient: float,
        method: PredictionMethod = PredictionMethod.DOT,
        **kwargs,
    ) -> np.array:
        """Recommends items to an unseen entity.

        Args:
            entity: A length-n vector of the new entity's ratings (TF SparseTensor or ndarray).
            c: Weight for negative training examples in the loss function,
                ie each positive example takes weight 1, while negative examples take
                discounted weight c.  Requires 0 < c < 1.
            regularization_coefficient: Coefficient on the embedding regularization
                term.
            method: The prediction method to use.

        Returns:
            An array of predicted values for the new entity.
        """
        if not (0 < c < 1):
            raise ValueError(f"c must satisfy 0 < c < 1 (got {c})")

        if hasattr(entity, "indices") and hasattr(entity, "values"):
            if not _TF_AVAILABLE:
                raise ImportError(
                    "TensorFlow required for SparseTensor entity input (not installed)"
                )
            dense_entity = tf.sparse.to_dense(tf.sparse.reorder(entity)).numpy()  # type: ignore[attr-defined]
        else:
            if not isinstance(entity, np.ndarray):
                raise TypeError(
                    "entity must be a TensorFlow SparseTensor or a numpy ndarray"
                )
            dense_entity = entity

        if dense_entity.shape != (self.n,):
            raise ValueError(
                f"entity shape mismatch: expected {(self.n,)}, got {dense_entity.shape}"
            )

        alpha = (1 / c) - 1

        new_entity_factor = self._update_factor(
            opposing_factors=self._V,
            data=np.expand_dims(dense_entity, axis=1),
            alpha=alpha,
            regularization_coefficient=regularization_coefficient,
        )

        if new_entity_factor.shape != (1, self._U.shape[1]):
            raise ValueError(
                "new_entity_factor shape mismatch: "
                f"expected {(1, self._U.shape[1])}, got {new_entity_factor.shape}"
            )

        return np.squeeze(
            calculate_predicted_matrix(new_entity_factor, self._V, method)
        )


Recommender.register(WalsRecommender)
