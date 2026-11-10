# SBM.py
from collections.abc import Callable
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy.special import betaln
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class SBMModel:
    """
    Bayesian SBM for undirected, binary networks without self-loops.

    Model:
        y_ij | xi, theta ~ Bernoulli(theta[xi_i, xi_j]), i < j
        theta_hk ~ Beta(a, b)

    The probabilities theta are integrated analytically.

    Parameters
    ----------
    adjacency : array-like
        Symmetric, binary adjacency matrix with a zero diagonal.
    a, b : float
        Positive parameters of the Beta prior on the connection probabilities
        between blocks.
    """

    adjacency: NDArray
    a: float = 1.0
    b: float = 1.0

    def __post_init__(self) -> None:
        Y = np.array(self.adjacency, dtype=np.float64, copy=True)
        if Y.ndim != 2 or Y.shape[0] != Y.shape[1]:
            raise ValueError("The adjacency matrix must be square.")
        if Y.shape[0] < 2:
            raise ValueError("The network must contain at least two nodes.")
        if not np.all(np.isfinite(Y)):
            raise ValueError("The matrix contains non-finite values.")
        if not np.allclose(Y, Y.T):
            raise ValueError("The matrix must be symmetric.")
        if not np.all((Y == 0) | (Y == 1)):
            raise ValueError("The matrix must contain only 0 and 1.")
        if not np.all(np.diag(Y) == 0):
            raise ValueError("Self-loops are not allowed: the diagonal must be zero.")
        if not np.isfinite(self.a) or not np.isfinite(self.b):
            raise ValueError("The parameters a and b must be finite.")
        if self.a <= 0 or self.b <= 0:
            raise ValueError("The parameters a and b must be positive.")
        Y.setflags(write=False)
        object.__setattr__(self, "adjacency", Y)
        rows, cols = np.triu_indices(Y.shape[0], k=1)
        object.__setattr__(self, "_rows", rows)
        object.__setattr__(self, "_cols", cols)
        object.__setattr__(self, "a", float(self.a))
        object.__setattr__(self, "b", float(self.b))
    @property
    def n_nodes(self) -> int:
        """Number of nodes in the network."""
        return self.adjacency.shape[0]
    @staticmethod
    def _canonicalize_partition(z: NDArray) -> IntArray:
        """
        Converts the labels to consecutive integers starting from zero.
        Example:
            [5, 5, 9, 9, 5] -> [0, 0, 1, 1, 0]
        """
        z = np.asarray(z)
        if z.ndim != 1:
            raise ValueError("The partition must be a one-dimensional array.")
        if not np.issubdtype(z.dtype, np.integer):
            raise ValueError("The labels must be integers.")
        if z.size == 0:
            raise ValueError("The partition cannot be empty.")
        _, inverse = np.unique(z, return_inverse=True)
        return inverse.astype(np.int64)

    def _validate_partition(self, z: NDArray) -> IntArray:
        """Validates the partition with respect to the number of nodes."""
        z = self._canonicalize_partition(z)
        if len(z) != self.n_nodes:
            raise ValueError(
                f"Expected {self.n_nodes} labels; "
                f"received {len(z)}."
            )
        return z

    
    def block_statistics(self, z: NDArray) -> dict[str, NDArray]:
        """
        Compute sufficient statistics by block.
        Each unordered pair is counted exactly once.
        """
        z = self._validate_partition(z)

        n_blocks = int(z.max()) + 1

        sizes = np.bincount(z, minlength=n_blocks)

        rows = self._rows
        cols = self._cols

        block_i = np.minimum(z[rows], z[cols])
        block_j = np.maximum(z[rows], z[cols])

        # Encode each block pair as a single integer.
        pair_index = block_i * n_blocks + block_j

        # Count edges using vectorized aggregation.
        edge_counts = np.bincount(
            pair_index,
            weights=self.adjacency[rows, cols],
            minlength=n_blocks**2
        ).reshape(n_blocks, n_blocks)

        # Convert to integer counts.
        edge_counts = edge_counts.astype(np.int64)

        # Recover the symmetric matrix.
        edges = edge_counts + np.triu(edge_counts, k=1).T

        # Number of possible edges in each block pair.
        possible = np.outer(sizes, sizes).astype(np.int64)

        diagonal = np.diag_indices(n_blocks)

        possible[diagonal] = sizes * (sizes - 1) // 2

        non_edges = possible - edges

        return {
            "labels": z,
            "sizes": sizes,
            "edges": edges,
            "non_edges": non_edges,
        }


    def log_marginal_likelihood(self, z: NDArray) -> float:
        """
        Compute log p(Y | xi), integrating out the theta probabilities.
        log p(Y | xi) =   sum_{h <= k} [log B(a + edges_hk, b + non_edges_hk) - log B(a, b)]
        """
        stats = self.block_statistics(z)
        edges = stats["edges"]
        non_edges = stats["non_edges"]
        upper = np.triu_indices_from(edges)
        log_prob = betaln(self.a + edges[upper], self.b + non_edges[upper]) - betaln(self.a, self.b)
        return float(np.sum(log_prob))

    def log_posterior(self, z: NDArray, log_prior: Callable[[IntArray], float] | None = None,) -> float:
        """
        Compute the log-posterior of the partition, up to a constant:
            log p(xi | Y) = log p(Y | xi) + log p(xi) + C.
        If log_prior is None, returns only log p(Y | xi).
        If provided, it should return log p(z) for the partition.
        """
        z = self._validate_partition(z)
        value = self.log_marginal_likelihood(z)
        if log_prior is not None:
            prior_value = float(log_prior(z.copy()))
            if not np.isfinite(prior_value):
                return prior_value
            value += prior_value
        return float(value)