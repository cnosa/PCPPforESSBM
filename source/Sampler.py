# Sampler.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from tqdm import tqdm
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
from .SBM import SBMModel
from .PartitionPrior import PartitionPrior


#-----------------------------------------------

@dataclass(frozen=True)
class SamplerConfig:
    n_iterations: int = 5_000
    burn_in: int = 1_000
    thin: int = 5

    def __post_init__(self):
        if self.n_iterations < 1:
            raise ValueError("n_iterations must be a positive integer.")
        if not 0 <= self.burn_in < self.n_iterations:
            raise ValueError("burn_in must be an integer between 0 and n_iterations.")
        if self.thin < 1:
            raise ValueError("thin must be a positive integer.")

#-----------------------------------------------


class BaseSampler(ABC):
    """Base class for MCMC samplers of SBM partitions."""

    def __init__(
        self,
        model: SBMModel,
        prior: PartitionPrior,
        config: SamplerConfig | None = None,
    ):
        self.model = model
        self.prior = prior
        self.config = config or SamplerConfig()

    @staticmethod
    def _canonicalize(z: NDArray) -> NDArray[np.int64]:
        """Renames blocks according to their first appearance."""
        z = np.asarray(z)

        if z.ndim != 1 or not np.issubdtype(z.dtype, np.integer):
            raise ValueError("The partition must be a vector of integers.")

        mapping = {}
        result = np.empty(len(z), dtype=np.int64)

        for i, label in enumerate(z):
            label = int(label)
            if label not in mapping:
                mapping[label] = len(mapping)
            result[i] = mapping[label]

        return result

    def log_target(self, z: NDArray) -> float:
        """Not noramlized log-posterior of the partition."""
        return self.model.log_posterior(z, log_prior=self.prior.log_probability,)

    @abstractmethod
    def _step(self, z: NDArray[np.int64], rng: np.random.Generator,) -> NDArray[np.int64]:
        """Executes a transition of the algorithm."""
        raise NotImplementedError

    def fit(self, initial_partition: NDArray, seed: int = 42, ) -> dict[str, object]:
        """Executes the chain and saves samples and traces."""
        z = self._canonicalize(initial_partition)
        if len(z) != self.model.n_nodes:
            raise ValueError("The partition must have a label for each node.")
        rng = np.random.default_rng(seed)
        samples = []
        log_posterior_trace = []
        n_communities_trace = []
        for iteration in tqdm(range(self.config.n_iterations),desc="Sampling"):
            z = self._canonicalize(self._step(z.copy(), rng))
            log_posterior_trace.append(self.log_target(z))
            n_communities_trace.append(len(np.unique(z)))

            if (iteration >= self.config.burn_in and (iteration - self.config.burn_in) % self.config.thin == 0):
                samples.append(z.copy())

        return {
            "samples": np.asarray(samples, dtype=np.int64),
            "log_posterior": np.asarray(log_posterior_trace),
            "n_communities": np.asarray(n_communities_trace),
            "final_partition": z.copy(),
            "seed": seed,
        }


#-----------------------------------------------

class GibbsSampler(BaseSampler):
    """Collapsed Gibbs sampler with node reassignment."""
    def _step(self, z: NDArray[np.int64], rng: np.random.Generator) -> NDArray[np.int64]:
        n = len(z)
        for i in rng.permutation(n):
            # Remove node i
            mask = np.arange(n) != i
            z_minus = self._canonicalize(z[mask])
            n_blocks = int(z_minus.max()) + 1
            log_weights = np.empty(n_blocks + 1, dtype=float)
            # Evaluate existing communities and one new community
            for k in range(n_blocks + 1):
                candidate = np.empty(n, dtype=np.int64)
                candidate[mask] = z_minus
                candidate[i] = k
                candidate = self._canonicalize(candidate)
                log_weights[k] = self.log_target(candidate)
            if np.all(np.isneginf(log_weights)):
                raise RuntimeError("All candidate assignments have zero posterior probability.")
            probabilities = np.exp(log_weights - logsumexp(log_weights))
            selected = rng.choice(n_blocks + 1, p=probabilities)
            z[mask] = z_minus
            z[i] = selected
            z = self._canonicalize(z)
        return z

#-----------------------------------------------

class SplitMergeSampler(BaseSampler):
    """
    MCMC split-merge for partitions of an SBM.
    If the chosen nodes are in the same block:
        propose to split it.
    If they are in different blocks:
        propose to merge them.
    The acceptance includes the Metropolis-Hastings correction.
    """
    def _step(self, z, rng):
        z = self._canonicalize(z.copy())
        n = len(z)
        # Choose two distinct nodes uniformly at random.
        i, j = rng.choice(n, size=2, replace=False)
        same_block = z[i] == z[j]
        if same_block:
            old_block = z[i]
            members = np.flatnonzero(z == old_block)
            candidate = z.copy()
            candidate[j] = z.max() + 1
            others = members[(members != i) & (members != j)]
            for node in others:
                if rng.random() < 0.5:
                    candidate[node] = candidate[i]
                else:
                    candidate[node] = candidate[j]
            candidate = self._canonicalize(candidate)
            log_q_forward = -len(others) * np.log(2.0)
            log_q_reverse = 0.0
        else:
            block_i = z[i]
            block_j = z[j]
            members_i = np.flatnonzero(z == block_i)
            members_j = np.flatnonzero(z == block_j)
            candidate = z.copy()
            candidate[candidate == block_j] = block_i
            candidate = self._canonicalize(candidate)
            n_others = len(members_i) + len(members_j) - 2
            log_q_forward = 0.0
            log_q_reverse = -n_others * np.log(2.0)
        # Aceptance rate of Metropolis-Hastings.
        log_ratio = self.log_target(candidate) - self.log_target(z) + log_q_reverse - log_q_forward
        if np.log(rng.random()) < min(0.0, log_ratio):
            return candidate
        return z

#-----------------------------------------------

class ChaperonesSampler(BaseSampler):
    """
    Chaperones Gibbs sampler for partitions of an SBM.
    Selects two anchor nodes and updates the nodes of the
    blocks that initially contained those anchors.
    Non-anchor nodes can only belong to a block
    that contains at least one of the anchors.
    """
    def _valid_state(self, z, anchors, union):
        """Checks that all nodes in union are still accompanied."""
        i, j = anchors
        anchor_labels = {z[i], z[j]}
        for node in union:
            if node not in anchors and z[node] not in anchor_labels:
                return False
        return True

    def _step(self, z, rng):
        z = self._canonicalize(np.asarray(z, dtype=np.int64).copy())
        n = len(z)
        if n < 2:
            raise ValueError("Chaperones requires at least two nodes.")
        # Select two anchor nodes, independently of the state.
        i, j = rng.choice(n, size=2, replace=False)
        anchors = (int(i), int(j))
        # Freeze the union of the original blocks of the anchors.
        initial_labels = {z[i], z[j]}
        union = np.flatnonzero(np.isin(z, list(initial_labels)))
        # Randomly update the nodes of that union.
        for node in rng.permutation(union):
            node = int(node)
            mask = np.arange(n) != node
            labels_in_union = np.unique(z[union[union != node]])
            if node in anchors:
                options = list(labels_in_union)
                options.append(int(z.max()) + 1)
            else:
                anchor_labels = {z[i], z[j]}
                options = [label for label in labels_in_union if label in anchor_labels]
            candidates = []
            log_weights = []
            seen = set()
            for label in options:
                candidate = z.copy()
                candidate[node] = label
                candidate = self._canonicalize(candidate)
                key = tuple(candidate)
                if key in seen:
                    continue
                seen.add(key)
                if not self._valid_state(candidate, anchors, union):
                    continue
                candidates.append(candidate)
                log_weights.append(self.log_target(candidate))
            if not candidates:
                raise RuntimeError("No valid candidate assignments were found.")
            log_weights = np.asarray(log_weights, dtype=float)
            if np.all(np.isneginf(log_weights)):
                raise RuntimeError("All candidates have zero posterior probability.")
            probabilities = np.exp(log_weights - logsumexp(log_weights))
            selected = rng.choice(len(candidates),p=probabilities)
            z = candidates[selected].copy()
        return z

#-----------------------------------------------