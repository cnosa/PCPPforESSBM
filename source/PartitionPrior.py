# PartitionPrior.py
from dataclasses import dataclass
from typing import Literal
import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln, logsumexp
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


PriorName = Literal["DP", "PY", "DM", "GN", "MFM", "PCPP"]
@dataclass(frozen=True)
class PartitionPrior:
    """
    Prior distributions over node partitions.

    Priors
    ------------------
    DP  : Dirichlet Process.
    PY  : Pitman-Yor Process.
    DM  : Dirichlet-Multinomial with H_total categories.
    GN  : Urn scheme de Gnedin.
    MFM : Mixture of Finite Mixtures.

    Parameters
    ----------
    kind : {"DP", "PY", "DM", "GN", "MFM"}
        Type of prior.
    alpha : float, optional
        Concentration parameter for DP and PY.
    sigma : float, optional
        Discount parameter for PY.
    beta : float, optional
        Category parameter for DM.
    H_total : int, optional
        Maximum number of categories for DM.
    gamma : float, optional
        Parameter of the GN scheme.
    rho : float, optional
        Symmetric Dirichlet parameter for MFM.
    p_k : tuple[float, ...], optional
        Probabilities P(K=m), for m=1,...,M.
        Automatically normalized.
    """

    kind: PriorName
    alpha: float | None = None
    sigma: float | None = None
    beta: float | None = None
    H_total: int | None = None
    gamma: float | None = None
    rho: float | None = None
    p_k: tuple[float, ...] | None = None
    theta: float | None = None
    omega: float | None = None
    pcpp_tol: float = 1e-10

    def __post_init__(self) -> None:
        kind = self.kind.upper()
        if kind not in {"DP", "PY", "DM", "GN", "MFM", "PCPP"}:
            raise ValueError("kind should be DP, PY, DM, GN, MFM or PCPP.")
        object.__setattr__(self, "kind", kind)
        if kind in {"DP", "PY"}:
            if self.alpha is None or not np.isfinite(self.alpha):
                raise ValueError("Requieres finite alpha.")
            if kind == "DP" and self.alpha <= 0:
                raise ValueError("For DP, alpha should be positive.")
        if kind == "PY":
            if self.sigma is None or not np.isfinite(self.sigma):
                raise ValueError("PY requires finite sigma.")
            if not 0 <= self.sigma < 1:
                raise ValueError("PY requires 0 <= sigma < 1.")
            if self.alpha <= -self.sigma:
                raise ValueError("PY requires alpha > -sigma.")
        if kind == "DM":
            if self.beta is None or not np.isfinite(self.beta) or self.beta <= 0:
                raise ValueError("DM requires beta positive and finite.")
            if (self.H_total is None
                or not isinstance(self.H_total, (int, np.integer))
                or self.H_total < 1):
                raise ValueError("DM requires H_total positive integer.")
        if kind == "GN":
            if self.gamma is None or not np.isfinite(self.gamma):
                raise ValueError("GN requires finite gamma.")
            if not 0 < self.gamma < 1:
                raise ValueError("GN requires 0 < gamma < 1.")
        if kind == "MFM":
            if self.rho is None or not np.isfinite(self.rho) or self.rho <= 0:
                raise ValueError("MFM requires rho positive and finite.")
            if self.p_k is None:
                raise ValueError("MFM requires specifying p_k.")
            p_k = np.asarray(self.p_k, dtype=float)

            if (p_k.ndim != 1
                or p_k.size == 0
                or np.any(~np.isfinite(p_k))
                or np.any(p_k < 0)
                or p_k.sum() <= 0):
                raise ValueError("p_k should be a vector of non-negative probabilities with positive sum.")
            p_k = p_k / p_k.sum()
            object.__setattr__(self, "p_k", tuple(p_k))
        if kind == "PCPP":
            if self.theta is None or not np.isfinite(self.theta):
                raise ValueError("PCPP requires finite theta.")
            if self.theta <= 0:
                raise ValueError("PCPP requires theta > 0.")
            if self.omega is None or not np.isfinite(self.omega):
                raise ValueError("PCPP requires finite omega.")
            if not 0 <= self.omega <= 1:
                raise ValueError("PCPP requires 0 <= omega <= 1.")
            if (not np.isfinite(self.pcpp_tol) or not 0 < self.pcpp_tol < 1):
                raise ValueError("PCPP requires 0 < pcpp_tol < 1.")
    @staticmethod
    def _canonicalize(z: NDArray) -> IntArray:
        """Rename the communities according to their first appearance."""
        z = np.asarray(z)
        if z.ndim != 1 or z.size == 0:
            raise ValueError("z should be a non-empty vector.")
        if not np.issubdtype(z.dtype, np.integer):
            raise ValueError("Labels must be integers.")
        mapping = {}
        canonical = np.empty(len(z), dtype=np.int64)
        for i, label in enumerate(z):
            label = int(label)
            if label not in mapping:
                mapping[label] = len(mapping)
            canonical[i] = mapping[label]
        return canonical

    @staticmethod
    def _sizes(z: IntArray) -> IntArray:
        """Returns the sizes of the non-empty communities."""
        return np.bincount(z)
    def _log_V(self, n: int, k: int) -> float:
        """
        Calculates log V_n(k) for MFM using log-sum-exp.
        p_k[m-1] = P(K=m), for m=1,...,M.
        """
        if self.kind != "MFM":
            raise ValueError("_log_V is only available for MFM.")
        if k < 1 or n < k:
            return -np.inf
        p_k = np.asarray(self.p_k, dtype=float)
        m_all = np.arange(1, len(p_k) + 1)
        valid = (m_all >= k) & (p_k > 0)
        if not np.any(valid):
            return -np.inf
        m = m_all[valid].astype(float)
        probs = p_k[valid]
        log_terms = gammaln(m + 1) - gammaln(m - k + 1) + gammaln(m * self.rho) - gammaln(m * self.rho + n) + np.log(probs)
        return float(logsumexp(log_terms))
    
    def _log_pcpp_conditional(self, sizes: IntArray, m: int,) -> float:
        """
        Log conditional EPPF of the Dirichlet approximation
        to the PCPP, given M = m.
        """
        n = int(sizes.sum())
        k = len(sizes)
        if m < k:
            return -np.inf
        omega = self.omega
        # Caso M = 1
        if m == 1:
            return 0.0 if k == 1 else -np.inf
        # Caso omega = 0: pesos uniformes
        if omega == 0:
            return float(gammaln(m + 1) - gammaln(m - k + 1) - n * np.log(m))
        # Parametro de la aproximacion Dirichlet
        alpha_m = m / omega**2 - 1 / m
        log_prob = (gammaln(m + 1) - gammaln(m - k + 1) + gammaln(m * alpha_m)
            - gammaln(m * alpha_m + n) + np.sum(gammaln(sizes + alpha_m) - gammaln(alpha_m)))
        return float(log_prob)
    def _log_pcpp_eppf(self, sizes: IntArray, ) -> float:
        """
        Marginal log-EPPF of the Dirichlet approximation
        to the PCPP.
        """
        n = int(sizes.sum())
        k = len(sizes)
        theta = self.theta
        # Truncation of M = 1 + Poisson(theta)
        from scipy.stats import poisson
        L = max(k, int(poisson.ppf(1 - self.pcpp_tol, theta)) + 1,)
        m_values = np.arange(k, L + 1)
        log_terms = []
        for m in m_values:
            log_p_m = (-theta + (m - 1) * np.log(theta) - gammaln(m))
            log_p_partition_given_m = (self._log_pcpp_conditional(sizes, m))
            log_terms.append(log_p_m + log_p_partition_given_m)
        return float(logsumexp(log_terms))
    def log_probability(self, z: NDArray) -> float:
        """
        Calculates the log-probability of a partition.
        """
        z = self._canonicalize(z)
        sizes = self._sizes(z)
        n = len(z)
        K = len(sizes)
        if self.kind == "DP":
            alpha = self.alpha
            return float(gammaln(alpha) - gammaln(alpha + n) + K * np.log(alpha) + np.sum(gammaln(sizes)))
        if self.kind == "PY":
            alpha = self.alpha
            sigma = self.sigma
            if K == 1:
                log_product = 0.0
            else:
                j = np.arange(1, K, dtype=float)
                terms = alpha + j * sigma
                if np.any(terms <= 0):
                    return -np.inf
                log_product = np.sum(np.log(terms))

            return float(log_product 
            + gammaln(alpha + 1) - gammaln(alpha + n) 
            + np.sum(gammaln(sizes - sigma) - gammaln(1 - sigma)))

        if self.kind == "DM":
            beta = self.beta
            H = self.H_total

            if K > H:
                return -np.inf

            log_injections = (
                gammaln(H + 1)
                - gammaln(H - K + 1)
            )

            return float(
                log_injections
                + gammaln(H * beta)
                - gammaln(n + H * beta)
                + np.sum(
                    gammaln(sizes + beta)
                    - gammaln(beta)
                )
            )

        if self.kind == "MFM":
            log_v = self._log_V(n, K)

            if not np.isfinite(log_v):
                return -np.inf

            rho = self.rho

            return float(log_v + np.sum(gammaln(sizes + rho) - gammaln(rho)))

        if self.kind == "PCPP":
            log_prob = self._log_pcpp_eppf(sizes)
            return float(log_prob)

        # GN
        gamma = self.gamma
        if not 0 < gamma < 1:
            raise ValueError("GN requires gamma in (0, 1).")
        z = np.asarray(z)
        n = len(z)
        if n == 0:
            return 0.0
        _, current_sizes = np.unique(z, return_counts=True)
        K = len(current_sizes)
        log_prob = (gammaln(K)
            + gammaln(K - gamma)
            - gammaln(1 - gamma)
            + gammaln(n - K + gamma)
            - gammaln(gamma)
            - gammaln(n)
            - gammaln(n + gamma)
            + gammaln(1 + gamma)
            + np.sum(gammaln(current_sizes + 1)))
        return float(log_prob)

    def urn_weights(self, sizes: NDArray) -> FloatArray:
        """
        Returns weights for existing communities and a new one.
        The last element corresponds to creating a new community.
        The weights are relative; predictive_probabilities() normalizes them.
        """
        sizes = np.asarray(sizes, dtype=np.float64)
        if sizes.ndim != 1 or sizes.size == 0 or np.any(sizes <= 0):
            raise ValueError("sizes must contain positive values.")
        K = len(sizes)
        n = int(sizes.sum())
        if self.kind == "DP":
            weights = np.append(sizes, self.alpha)
        elif self.kind == "PY":
            weights = np.append(sizes - self.sigma, self.alpha + K * self.sigma,)
        elif self.kind == "DM":
            if K > self.H_total:
                raise ValueError("There are more communities than H_total.")
            weights = np.append(sizes + self.beta,self.beta * (self.H_total - K),)

        elif self.kind == "MFM":
            log_v_n_k = self._log_V(n, K)

            if not np.isfinite(log_v_n_k):
                raise ValueError(
                    "The partition has zero prior probability under MFM."
                )

            log_v_next_k = self._log_V(n + 1, K)
            log_v_next_k1 = self._log_V(n + 1, K + 1)

            existing_ratio = (
                0.0
                if not np.isfinite(log_v_next_k)
                else np.exp(log_v_next_k - log_v_n_k)
            )

            new_ratio = (
                0.0
                if not np.isfinite(log_v_next_k1)
                else np.exp(log_v_next_k1 - log_v_n_k)
            )

            weights = np.append(
                (sizes + self.rho) * existing_ratio,
                self.rho * new_ratio,
            )

        elif self.kind == "PCPP":

            log_p_current = self._log_pcpp_eppf(
                sizes.astype(np.int64)
            )

            # Existing communities
            existing_weights = np.empty(K, dtype=float)

            for r in range(K):

                new_sizes = sizes.astype(np.int64).copy()
                new_sizes[r] += 1

                log_p_new = self._log_pcpp_eppf(new_sizes)

                existing_weights[r] = np.exp(
                    log_p_new - log_p_current
                )

            # New community
            new_sizes = np.append(
                sizes.astype(np.int64),
                1,
            )

            log_p_new = self._log_pcpp_eppf(new_sizes)

            new_weight = np.exp(
                log_p_new - log_p_current
            )

            weights = np.append(
                existing_weights,
                new_weight,
            )

        else:  # GN
            gamma = self.gamma

            existing = (sizes + 1) * (n - K + gamma)
            new = K**2 - K * gamma

            weights = np.append(existing, new)

        if np.any(~np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("The prior produced invalid weights.")

        if weights.sum() <= 0:
            raise ValueError("The sum of the weights must be positive.")

        return weights.astype(np.float64)

    def predictive_probabilities(self, sizes: NDArray) -> FloatArray:
        """Normalizes the weights to obtain predictive probabilities."""
        weights = self.urn_weights(sizes)
        return weights / weights.sum()