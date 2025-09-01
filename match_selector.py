from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, List, Sequence, Tuple

try:  # Optional dependency
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - executed when numpy missing
    np = None  # type: ignore

try:  # stable-baselines3 is optional when using RL helpers
    from stable_baselines3.common.base_class import BaseAlgorithm
except Exception:  # pragma: no cover - only executed if library missing
    BaseAlgorithm = Any


@dataclass(frozen=True)
class MatchPrediction:
    """Single match prediction with probability of success.

    Attributes:
        name: Human readable name of the match.
        start_time: Kick-off time. It is kept for reference but not used in
            ordering.
        probability: Estimated probability of the selected outcome (0-1).
    """

    name: str
    start_time: datetime
    probability: float

    def __str__(self) -> str:
        """Human-readable representation including kick-off time."""
        kickoff = self.start_time.strftime("%Y-%m-%d %H:%M")
        return f"{self.name} ({kickoff}): {self.probability:.0%}"


def rank_by_probability(matches: Iterable[MatchPrediction]) -> List[MatchPrediction]:
    """Return matches sorted descending by probability.

    Args:
        matches: Iterable of :class:`MatchPrediction`.

    Returns:
        A list of matches ordered from most to least probable. Date and time do
        not influence the ranking.
    """

    return sorted(matches, key=lambda m: m.probability, reverse=True)


def max_series(
    matches: Iterable[MatchPrediction], *, limit: int | None = None
) -> List[MatchPrediction]:
    """Return matches ordered to maximise a potential streak.

    The function simply ranks matches by probability, ignoring kick-off times
    or any other metadata. Optionally the result can be truncated to ``limit``
    items which allows selecting only the top-N predictions for a coupon.

    Args:
        matches: Iterable of :class:`MatchPrediction`.
        limit: Maximum number of matches to return. ``None`` keeps all.

    Returns:
        A list of matches ordered from most to least probable.
    """

    ranked = rank_by_probability(matches)
    return ranked if limit is None else ranked[:limit]


def longest_confident_series(
    matches: Iterable[MatchPrediction], *, min_probability: float = 0.5
) -> List[MatchPrediction]:
    """Select a sequence of high-confidence matches.

    The function ranks all matches by probability and then keeps only those
    whose probability is above ``min_probability``. The order is purely based on
    probability, independent of match dates.

    Args:
        matches: Iterable of :class:`MatchPrediction`.
        min_probability: Minimum probability required to keep a match in the
            returned series.

    Returns:
        List of matches forming a series with the highest confidence first.
    """

    ranked = rank_by_probability(matches)
    return [m for m in ranked if m.probability >= min_probability]


def predictions_from_rl(
    model: BaseAlgorithm,
    observations: Sequence[np.ndarray],
    meta: Sequence[Tuple[str, datetime]],
    *,
    action_index: int = 0,
) -> List[MatchPrediction]:
    """Convert RL policy outputs into :class:`MatchPrediction` objects.

    Args:
        model: Trained RL model compatible with ``stable_baselines3``.
        observations: Sequence of environment observations for each match.
        meta: Sequence of ``(name, start_time)`` pairs describing matches.
        action_index: Index of the action interpreted as a "bet".

    Returns:
        List of :class:`MatchPrediction` with probabilities derived from the
        model's action distribution.
    """

    preds: List[MatchPrediction] = []
    for obs, (name, start_time) in zip(observations, meta):
        if np is None:
            raise ImportError("numpy is required for predictions_from_rl")
        obs_arr = np.asarray(obs)[None, ...]
        dist = model.policy.get_distribution(obs_arr)
        probs = getattr(dist.distribution, "probs", None)
        if probs is None:
            logits = dist.distribution.logits
            probs = logits.softmax(-1)
        prob = float(probs[0][action_index])
        preds.append(MatchPrediction(name, start_time, prob))
    return preds


def rl_confident_series(
    model: BaseAlgorithm,
    observations: Sequence[np.ndarray],
    meta: Sequence[Tuple[str, datetime]],
    *,
    min_probability: float = 0.5,
    action_index: int = 0,
) -> List[MatchPrediction]:
    """Build a confident series using an RL model for probabilities."""

    matches = predictions_from_rl(model, observations, meta, action_index=action_index)
    return longest_confident_series(matches, min_probability=min_probability)


def rl_max_series(
    model: BaseAlgorithm,
    observations: Sequence[np.ndarray],
    meta: Sequence[Tuple[str, datetime]],
    *,
    limit: int | None = None,
    action_index: int = 0,
) -> List[MatchPrediction]:
    """Build the highest-confidence streak using an RL model.

    This helper converts the model outputs into ``MatchPrediction`` objects and
    then orders them purely by probability. The kick-off times are ignored so
    the returned list represents the best theoretical series regardless of
    chronology.

    Args:
        model: Trained RL model compatible with ``stable_baselines3``.
        observations: Sequence of environment observations for each match.
        meta: Sequence of ``(name, start_time)`` pairs describing matches.
        limit: Optional maximum number of matches to return.
        action_index: Index of the action interpreted as a "bet".

    Returns:
        List of matches ordered by descending probability.
    """

    matches = predictions_from_rl(model, observations, meta, action_index=action_index)
    return max_series(matches, limit=limit)


if __name__ == "__main__":
    sample_matches = [
        MatchPrediction("Team A vs Team B", datetime(2025, 6, 1, 18, 0), 0.72),
        MatchPrediction("Team C vs Team D", datetime(2025, 6, 2, 21, 0), 0.65),
        MatchPrediction("Team E vs Team F", datetime(2025, 5, 30, 16, 0), 0.81),
    ]

    print("Top two matches by confidence:")
    for match in max_series(sample_matches, limit=2):
        print(match)

    print("\nMatches above 60% probability:")
    for match in longest_confident_series(sample_matches, min_probability=0.6):
        print(match)
