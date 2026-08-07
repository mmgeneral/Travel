"""Feature-indexed Beta-Bernoulli belief store for checkpoint scheduling (v1)."""

from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

from slot_model import Slot


class BeliefStore:
    """Tracks beta parameters for tags and tag pairs, plus shrinkage combination.

    ``[SIMPLIFIED]``: only pairwise interactions are maintained, and slots with
    3+ tags are combined using a geometric-mean heuristic (see
    :meth:`combine_multi_tag_probability`).  No DB is used; persistence is a
    simple JSON file.
    """

    def __init__(self, path: Optional[str] = None, kappa: float = 5.0):
        self.kappa = kappa
        self.feature_beta: Dict[str, Tuple[float, float]] = {}
        self.pair_beta: Dict[FrozenSet[str], Tuple[float, float]] = {}
        self.path = Path(path) if path else None
        if self.path and self.path.exists():
            self.load(self.path)

    # -- basic helpers ------------------------------------------------------

    def _ensure_tag(self, tag: str) -> None:
        if tag not in self.feature_beta:
            self.feature_beta[tag] = (1.0, 1.0)

    def _ensure_pair(self, pair: FrozenSet[str]) -> None:
        if pair not in self.pair_beta:
            self.pair_beta[pair] = (1.0, 1.0)

    @staticmethod
    def _pair_key(pair: Iterable[str]) -> FrozenSet[str]:
        return frozenset(pair)

    # -- update methods -----------------------------------------------------

    def record_failure(self, slot: Slot) -> None:
        """Increments beta (failure count) for all tags and tag pairs."""
        self._update(slot, success=False)

    def record_success(self, slot: Slot) -> None:
        """Increments alpha (success count) for all tags and tag pairs."""
        self._update(slot, success=True)

    def record_correction(self, slot: Slot) -> None:
        """Alias for :meth:`record_failure` — a Layer‑1 user correction is a beta increment."""
        self.record_failure(slot)

    def _update(self, slot: Slot, success: bool) -> None:
        # unique tags preserve insertion order
        tags = list(dict.fromkeys(t for t in slot.risk_tags if t))
        if not tags:
            return

        for t in tags:
            self._ensure_tag(t)
            a, b = self.feature_beta[t]
            if success:
                self.feature_beta[t] = (a + 1.0, b)
            else:
                self.feature_beta[t] = (a, b + 1.0)

        for pair in combinations(tags, 2):
            key = self._pair_key(pair)
            self._ensure_pair(key)
            a, b = self.pair_beta[key]
            if success:
                self.pair_beta[key] = (a + 1.0, b)
            else:
                self.pair_beta[key] = (a, b + 1.0)

    # -- probability calculation -------------------------------------------

    def probability_for_tags(self, tags: List[str]) -> float:
        """Combined success probability for a slot given its risk tags."""
        unique_tags = list(dict.fromkeys(t for t in tags if t))
        if not unique_tags:
            return 1.0
        if len(unique_tags) == 1:
            return self._marginal_prob(unique_tags[0])
        if len(unique_tags) == 2:
            return self._pair_shrink(unique_tags)
        return self.combine_multi_tag_probability(unique_tags)

    def _marginal_prob(self, tag: str) -> float:
        self._ensure_tag(tag)
        a, b = self.feature_beta[tag]
        denom = a + b
        return a / denom if denom > 0 else 1.0

    def _pair_shrink(self, tags: List[str]) -> float:
        """Shrinkage combination for exactly two tags."""
        a_tag, b_tag = tags[0], tags[1]
        p_a = self._marginal_prob(a_tag)
        p_b = self._marginal_prob(b_tag)
        key = self._pair_key(tags)
        self._ensure_pair(key)
        a_pair, b_pair = self.pair_beta[key]
        n_pair = a_pair + b_pair - 2.0  # observed pair count (prior pseudo‑count subtracted)
        w = n_pair / (n_pair + self.kappa)
        p_pair = a_pair / (a_pair + b_pair) if (a_pair + b_pair) > 0 else 1.0
        return w * p_pair + (1.0 - w) * (p_a * p_b)

    def combine_multi_tag_probability(self, tags: List[str]) -> float:
        """``[SIMPLIFIED]`` heuristic for slots with 3+ tags.

        Computes the shrink probability for every tag pair and returns the
        geometric mean of those pair probabilities.  This replaces a full
        multi‑way model and is intentionally simple for the v1 prototype.
        """
        pair_probs = []
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                pair_probs.append(self._pair_shrink([tags[i], tags[j]]))
        if not pair_probs:
            return 1.0
        log_sum = sum(math.log(p) for p in pair_probs)
        return math.exp(log_sum / len(pair_probs))

    # -- persistence --------------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        target = Path(path) if path else self.path
        if target is None:
            raise ValueError("No path specified for belief store.")
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "feature_beta": {k: list(v) for k, v in self.feature_beta.items()},
            "pair_beta": {
                "|".join(sorted(key)): list(value)
                for key, value in self.pair_beta.items()
            },
        }
        target.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, path: str) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.feature_beta = {
            k: tuple(v) for k, v in data.get("feature_beta", {}).items()
        }
        self.pair_beta = {}
        for key, value in data.get("pair_beta", {}).items():
            parts = key.split("|")
            self.pair_beta[frozenset(parts)] = tuple(value)
