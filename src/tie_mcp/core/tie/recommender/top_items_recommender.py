from typing import Any

import numpy as np
from sklearn.metrics import mean_squared_error

from .recommender import Recommender


class TopItemsRecommender(Recommender):
    """A recommender model which always recommends the most observed techniques.

    A recommender model which always recommends the most observed techniques in the
    dataset in frequency order.
    """

    # Abstraction function:
    #   AF(m, n, item_frequencies) = a recommender model which recommends the n
    #       items in order of frequency according to item_frequencies
    #       for each of the m entities
    # Rep invariant:
    #   - m > 0
    #   - n > 0
    #   - item_frequencies.shape == (n,)
    #   - 0 <= item_frequencies[i] <= n-1 for all 0 <= i < n
    # Safety from rep exposure:
    #   - m and n are private and immutable
    #   - item_frequency is private and never returned

    def __init__(self, m, n, k):
        """Initializes a TopItemsRecommender object."""
        if m <= 0:
            raise ValueError(f"m must be > 0 (got {m})")
        if n <= 0:
            raise ValueError(f"n must be > 0 (got {n})")
        # k unused but kept for signature consistency with other recommenders
        self._m = m  # entity dimension
        self._n = n  # item dimension

        # array of item frequencies,
        # ranging from 0 (least frequent) to n-1 (most frequent)
        self._item_frequencies = np.zeros((n,))

        self._checkrep()

    def _checkrep(self):
        """Validates the rep invariant; raises ValueError on violation."""
        if self._m <= 0:
            raise ValueError(f"Invalid state: m must be > 0 (got {self._m})")
        if self._n <= 0:
            raise ValueError(f"Invalid state: n must be > 0 (got {self._n})")
        if self._item_frequencies.shape != (self._n,):
            raise ValueError(
                "item_frequencies shape mismatch: "
                f"expected {(self._n,)}, got {self._item_frequencies.shape}"
            )
        if not (0 <= self._item_frequencies).all():
            raise ValueError("Item frequencies contain negative values")
        if not (self._item_frequencies <= self._n - 1).all():
            raise ValueError(
                "Item frequencies contain values greater than allowed upper bound"
            )

    def U(self) -> np.ndarray:
        """Gets U as a factor of the factorization UV^T."""
        raise NotImplementedError

    def V(self) -> np.ndarray:
        """Gets V as a factor of the factorization UV^T."""
        raise NotImplementedError

    def _scale_item_frequency(self, item_frequencies: np.array) -> np.array:
        """Scales the item frequencies from 0 to 1.

        Assigns each item the value 1/(n-1) * rank_i, where rank_i is the rank
        of item i in sorted ascending order by frequency.
        Therefore, the top frequency item will take scaled value 1, while the least
        frequent item will take scaled value 0.

        Args:
            item_frequencies: A length-n vector containing the number of occurrences
                of each item in the dataset.

        Returns:
            A scaled version of item_frequencies.
        """
        # validate 1d array
        if len(item_frequencies.shape) != 1:
            raise ValueError(
                f"item_frequencies must be 1D (got shape {item_frequencies.shape})"
            )

        scaled_ranks = item_frequencies / (len(item_frequencies) - 1)

        if scaled_ranks.shape != item_frequencies.shape:
            raise ValueError(
                "Scaled ranks shape mismatch: "
                f"expected {item_frequencies.shape}, got {scaled_ranks.shape}"
            )

        self._checkrep()
        return scaled_ranks

    def fit(self, data: Any, **kwargs):
        """Fit by computing item frequency ranks.

        Accepts either:
          - A TensorFlow SparseTensor (if tensorflow installed)
          - A dense numpy ndarray of shape (m, n)
        """
        if hasattr(data, "indices"):  # likely a TF sparse tensor
            try:  # pragma: no cover - optional dependency
                import tensorflow as tf  # type: ignore
                technique_matrix: np.ndarray = tf.sparse.to_dense(
                    tf.sparse.reorder(data)
                ).numpy()
            except Exception as e:  # pragma: no cover
                raise ImportError(
                    "TensorFlow required to pass a SparseTensor to TopItemsRecommender.fit()"
                ) from e
        else:
            if not isinstance(data, np.ndarray):
                raise TypeError(
                    "data must be a TensorFlow SparseTensor or a numpy ndarray"
                )
            technique_matrix = data

        technique_frequency = technique_matrix.sum(axis=0)
        if technique_frequency.shape != (self._n,):
            raise ValueError(
                "Technique frequency shape mismatch: "
                f"expected {(self._n,)}, got {technique_frequency.shape}"
            )

        ranks = technique_frequency.argsort().argsort()

        self._item_frequencies = ranks
        self._checkrep()

    def evaluate(self, test_data: Any, **kwargs) -> float:
        """Evaluate using MSE against observed entries in test_data.

        test_data may be:
          - TensorFlow SparseTensor
          - numpy ndarray (dense) of shape (m, n)
        """
        predictions_matrix = self.predict()

        if hasattr(test_data, "indices"):
            td = test_data
            row_indices = tuple(index[0] for index in td.indices)
            column_indices = tuple(index[1] for index in td.indices)
            prediction_values = predictions_matrix[row_indices, column_indices]
            values = td.values
        else:
            if not isinstance(test_data, np.ndarray):
                raise TypeError(
                    "test_data must be a TensorFlow SparseTensor or a numpy ndarray"
                )
            if test_data.shape != (self._m, self._n):
                raise ValueError(
                    f"test_data shape mismatch: expected {(self._m, self._n)}, got {test_data.shape}"
                )
            prediction_values = predictions_matrix[test_data > 0]
            values = test_data[test_data > 0]

        self._checkrep()
        return mean_squared_error(values, prediction_values)

    def predict(self, **kwargs) -> np.ndarray:
        scaled_ranks = self._scale_item_frequency(self._item_frequencies)
        matrix = np.repeat(np.expand_dims(scaled_ranks, axis=1), self._m, axis=1).T

        if matrix.shape != (self._m, self._n):
            raise RuntimeError(
                "Prediction matrix shape mismatch: "
                f"expected {(self._m, self._n)}, got {matrix.shape}"
            )

        self._checkrep()
        return matrix

    def predict_new_entity(self, entity: Any, **kwargs) -> np.array:
        """Predict scores for a new entity (identical ranking)."""
        self._checkrep()
        return self._scale_item_frequency(self._item_frequencies)
