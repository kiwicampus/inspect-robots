"""Scoring: Scores, the Scorer protocol, epoch reducers, and builtin scorers.

Mirrors Inspect AI's ``@scorer``/reducer split. A scorer maps a recorded
trajectory (+ the scene's ``Target``) to a [`Score`][inspect_robots.scorer.Score]; an epoch
*reducer* collapses the per-epoch scores of one scene into a single score before metrics
aggregate across scenes.

Scorers consume the *recorded* trajectory (not a live environment), so scoring is
reproducible from a saved log.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from math import asin, comb, cos, radians, sin, sqrt
from statistics import mean as _mean
from statistics import median as _median
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from inspect_robots.scene import Target

if TYPE_CHECKING:
    from inspect_robots.rollout import TrialRecord

ScoreValue = bool | int | float | str
Reducer = Callable[[Sequence["Score"]], "Score"]


@dataclass(frozen=True)
class Score:
    """The outcome a scorer assigns to one trajectory."""

    value: ScoreValue
    explanation: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def value_to_float(value: ScoreValue) -> float:
    """Coerce a score value to a float for metric aggregation."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(value)
    except ValueError:
        return 0.0


@runtime_checkable
class Scorer(Protocol):
    """Maps a recorded trajectory + scene target to a [`Score`][inspect_robots.scorer.Score]."""

    @property
    def name(self) -> str:
        """Stable identifier used as the score key in evaluation results."""
        ...

    def __call__(self, record: TrialRecord, target: Target | None) -> Score:
        """Assign one score from the recorded trajectory and optional target."""
        ...


# --------------------------------------------------------------------------- #
# Epoch reducers: list[Score] -> Score  (namespaced separately from metrics)
# --------------------------------------------------------------------------- #
def _numeric(value: ScoreValue) -> float:
    """Strictly coerce a value to a number for numeric reduction.

    Unlike [`value_to_float`][inspect_robots.scorer.value_to_float] (which is lenient for metric
    aggregation), this
    raises on a non-numeric string rather than silently coercing it to 0.0 — so a
    ``mean`` over categorical scores fails loudly instead of lying.
    """
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(value)
    except ValueError:
        raise TypeError(
            f"cannot numerically reduce non-numeric score value {value!r}; "
            "use a categorical reducer such as 'mode'"
        ) from None


def reduce_mean(scores: Sequence[Score]) -> Score:
    """Collapse numeric epoch values with the arithmetic mean."""
    return Score(value=_mean(_numeric(s.value) for s in scores))


def reduce_median(scores: Sequence[Score]) -> Score:
    """Collapse numeric epoch values with the median."""
    return Score(value=_median(_numeric(s.value) for s in scores))


def reduce_max(scores: Sequence[Score]) -> Score:
    """Keep the largest numeric epoch value."""
    return Score(value=max(_numeric(s.value) for s in scores))


def reduce_min(scores: Sequence[Score]) -> Score:
    """Keep the smallest numeric epoch value."""
    return Score(value=min(_numeric(s.value) for s in scores))


def reduce_mode(scores: Sequence[Score]) -> Score:
    """Most common raw value (works for categorical scores). Deterministic."""
    values = [s.value for s in scores]
    counts = Counter(values)
    best = max(values, key=lambda v: (counts[v], str(v)))
    return Score(value=best)


def pass_at_k(k: int) -> Reducer:
    """Unbiased pass@k estimator over the epoch scores (success = value >= 0.5)."""
    if k < 1:
        raise ValueError("k must be >= 1")

    def reducer(scores: Sequence[Score]) -> Score:
        n = len(scores)
        c = sum(1 for s in scores if _numeric(s.value) >= 0.5)
        if k > n:
            raise ValueError(f"pass_at_{k} needs at least {k} epochs, got {n}")
        # 1 - C(n-c, k) / C(n, k): probability >=1 of k draws is correct.
        value = 1.0 - (comb(n - c, k) / comb(n, k) if n - c >= k else 0.0)
        return Score(value=value)

    return reducer


_REDUCERS: dict[str, Reducer] = {
    "mean": reduce_mean,
    "median": reduce_median,
    "max": reduce_max,
    "min": reduce_min,
    "mode": reduce_mode,
}


def get_reducer(name: str) -> Reducer:
    """Resolve a builtin reducer or parse a dynamic ``pass_at_<k>`` name."""
    if name in _REDUCERS:
        return _REDUCERS[name]
    if name.startswith("pass_at_"):
        try:
            return pass_at_k(int(name[len("pass_at_") :]))
        except ValueError as exc:
            raise ValueError(f"invalid pass@k reducer {name!r}: {exc}") from None
    raise ValueError(f"unknown epoch reducer {name!r}; known: {sorted(_REDUCERS)} or 'pass_at_<k>'")


def reduce_scores(name: str, scores: Sequence[Score]) -> Score:
    """Apply the named epoch reducer to one scene's scores."""
    return get_reducer(name)(scores)


# --------------------------------------------------------------------------- #
# Builtin scorers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _SuccessAtEnd:
    name: str = "success_at_end"

    def __call__(self, record: TrialRecord, target: Target | None) -> Score:
        last = record.steps[-1] if record.steps else None
        success = bool(
            last is not None
            and last.result.terminated
            and last.result.termination_reason == "success"
        )
        return Score(
            value=success,
            explanation="reached success termination" if success else "did not succeed",
        )


def success_at_end() -> Scorer:
    """Score 1.0 iff the episode terminated with reason ``"success"``."""
    return _SuccessAtEnd()


@dataclass(frozen=True)
class _EpisodeLength:
    name: str = "episode_length"

    def __call__(self, record: TrialRecord, target: Target | None) -> Score:
        return Score(value=len(record.steps))


def episode_length() -> Scorer:
    """Score = number of environment steps taken."""
    return _EpisodeLength()


def _distances(record: TrialRecord) -> list[float]:
    return [float(s.result.info["distance"]) for s in record.steps if "distance" in s.result.info]


@dataclass(frozen=True)
class _MinDistanceToGoal:
    name: str = "min_distance_to_goal"

    def __call__(self, record: TrialRecord, target: Target | None) -> Score:
        dists = _distances(record)
        if not dists:
            return Score(value=float("inf"), explanation="no distance signal recorded")
        return Score(value=min(dists))


def min_distance_to_goal() -> Scorer:
    """Score = the closest the effector got to the goal (lower is better)."""
    return _MinDistanceToGoal()


@dataclass(frozen=True)
class _ReachedGoalState:
    threshold: float
    name: str = "reached_goal_state"

    def __call__(self, record: TrialRecord, target: Target | None) -> Score:
        dists = _distances(record)
        reached = bool(dists) and min(dists) <= self.threshold
        return Score(value=reached, explanation=f"min_distance <= {self.threshold}")


def reached_goal_state(threshold: float = 0.05) -> Scorer:
    """Success iff the effector came within ``threshold`` of the goal."""
    return _ReachedGoalState(threshold=threshold)


_EARTH_RADIUS_M = 6371008.8  # IUGG mean radius


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters between two (lat, lon) points, degrees in.

    No third-party dependency and no NumPy: core stays NumPy-only, and this
    doesn't even need an array. Accurate to well under a meter at the scale a
    ground robot's GPS error already dominates.
    """
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    return 2 * _EARTH_RADIUS_M * asin(min(1.0, sqrt(a)))


@dataclass(frozen=True)
class _DistanceToGoal:
    """Real-hardware analog of ``min_distance_to_goal``: no privileged sim ``info``
    is available on real embodiments, so this reads recorded GPS out of the
    trajectory's own ``Observation.state`` instead (R6: a pure reader of the log).
    """

    state_key: str = "gps"
    name: str = "distance_to_goal"

    def __call__(self, record: TrialRecord, target: Target | None) -> Score:
        if target is None or "goal_lat" not in target.spec or "goal_lon" not in target.spec:
            return Score(value=float("inf"), explanation="no goal declared in Target.spec")
        goal_lat = float(target.spec["goal_lat"])
        goal_lon = float(target.spec["goal_lon"])
        for step in reversed(record.steps):
            state = step.observation.state.get(self.state_key)
            if state is None or len(state) < 2:
                continue
            dist = _haversine_m(float(state[0]), float(state[1]), goal_lat, goal_lon)
            return Score(value=dist, explanation=f"{dist:.2f} m from goal at trial end")
        return Score(
            value=float("inf"),
            explanation=f"no {self.state_key!r} state recorded in this trial",
        )


def distance_to_goal(state_key: str = "gps") -> Scorer:
    """Great-circle distance (meters) from the trial's last recorded GPS fix to
    ``Target.spec``'s ``goal_lat``/``goal_lon`` (degrees). Lower is better.

    Reads ``Observation.state[state_key]`` as ``[latitude, longitude, ...]``
    (extra entries, e.g. a compass heading, are ignored) from the *last* step
    that recorded it, so a state field an embodiment only fills in
    intermittently still scores correctly. Returns ``inf`` (never raises) when
    the scene has no goal, or the trial has no such state recorded at all --
    both are configuration gaps worth seeing in a result, not a crash.
    """
    return _DistanceToGoal(state_key=state_key)


# Recognized affirmative operator verdicts (case-insensitive).
_OPERATOR_SUCCESS = frozenset({"success", "pass", "yes", "y", "1", "true"})


def is_affirmative_verdict(verdict: str | None) -> bool:
    """Whether a recorded operator judgement reads as "the trial succeeded".

    The single public definition of that contract: the recognized affirmative
    vocabulary *and* the comparison rules around it (surrounding whitespace is
    ignored, matching is case-insensitive, and ``None`` — no judgement recorded
    — is not affirmative). Benchmarks that grade real-world runs from operator
    verdicts should call this rather than restate any part of it, so a change
    here reaches every consumer at once.
    """
    return verdict is not None and verdict.strip().lower() in _OPERATOR_SUCCESS


@dataclass(frozen=True)
class _OperatorScorer:
    name: str = "operator"

    def __call__(self, record: TrialRecord, target: Target | None) -> Score:
        # R6: the human verdict is captured once during rollout and recorded;
        # this scorer only READS it, so scoring stays reproducible from a log.
        verdict = record.operator_judgement
        if verdict is None:
            return Score(value=False, explanation="no operator judgement recorded")
        return Score(
            value=is_affirmative_verdict(verdict),
            explanation=f"operator verdict: {verdict!r}",
        )


def operator_scorer() -> Scorer:
    """Score from the human operator's recorded success judgement (R6)."""
    return _OperatorScorer()


def _gps_trace(
    record: TrialRecord, state_key: str
) -> tuple[tuple[float, float] | None, float, float, float]:
    """The first recorded ``(lat, lon)`` for ``state_key``, that fix's
    ``state_time``, the *last* recorded fix's ``state_time``, and the summed
    great-circle length (meters) of every consecutive recorded fix, in step
    order.

    Steps where the state is absent or too short are skipped rather than
    breaking the chain, so a field an embodiment only fills in intermittently
    still contributes a connected polyline through the fixes it did record
    (same tolerance as ``distance_to_goal``). Returns ``(None, 0.0, 0.0, 0.0)``
    when the state was never recorded at all.
    """
    first: tuple[float, float] | None = None
    first_time = 0.0
    last_time = 0.0
    prev: tuple[float, float] | None = None
    total = 0.0
    for step in record.steps:
        state = step.observation.state.get(state_key)
        if state is None or len(state) < 2:
            continue
        point = (float(state[0]), float(state[1]))
        if first is None:
            first = point
            first_time = step.observation.state_time
        last_time = step.observation.state_time
        if prev is not None:
            total += _haversine_m(prev[0], prev[1], point[0], point[1])
        prev = point
    return first, first_time, last_time, total


@dataclass(frozen=True)
class _SuccessWeightedByPathLength:
    """Success weighted by Path Length (Anderson et al. 2018): per-episode
    ``S_i * (l_i / max(p_i, l_i))``.

    Real-hardware analog of the metric: no geodesic path planner is available
    on this stack, so both terms are derived from what's actually recorded:

    - ``S_i``: the recorded operator verdict (``is_affirmative_verdict`` of
      ``record.operator_judgement``) — the same success signal
      ``operator_scorer`` reports (R6: one recorded human judgement is the
      single source of "did this actually succeed", not a second,
      GPS-proximity-based notion of success).
    - ``p_i``: the summed great-circle length of the trial's recorded
      ``state_key`` fixes.
    - ``l_i``: ``Target.spec["optimal_path_length"]`` (meters) when given —
      e.g. supplied by a real path-planning/map tool — else the straight-line
      (haversine) distance from the trial's first recorded fix to
      ``Target.spec["goal_lat"]``/``"goal_lon"]``. The fallback is an honest
      simplification: it under-estimates the true optimal path whenever the
      real route can't be a straight line (obstacles, terrain), which makes
      this SPL a conservative (never-inflated) estimate of the textbook
      metric.

    Degenerate cases: no ``state_key`` ever recorded, or no way to determine
    ``l_i`` (no ``Target``, or a ``Target.spec`` with neither
    ``optimal_path_length`` nor both goal fields) both score ``0.0`` rather
    than raising. ``l_i == p_i == 0`` (already at the goal, no movement
    recorded) scores ``1.0`` by convention when successful, avoiding a 0/0
    divide.
    """

    state_key: str = "gps"
    name: str = "spl"

    def __call__(self, record: TrialRecord, target: Target | None) -> Score:
        first, _first_time, _last_time, path_m = _gps_trace(record, self.state_key)
        if first is None:
            return Score(
                value=0.0,
                explanation=f"no {self.state_key!r} state recorded in this trial",
            )

        optimal_m: float | None = None
        if target is not None and "optimal_path_length" in target.spec:
            optimal_m = float(target.spec["optimal_path_length"])
        elif target is not None and "goal_lat" in target.spec and "goal_lon" in target.spec:
            optimal_m = _haversine_m(
                first[0], first[1], float(target.spec["goal_lat"]), float(target.spec["goal_lon"])
            )
        if optimal_m is None:
            return Score(
                value=0.0,
                explanation="no optimal_path_length or goal_lat/goal_lon in Target.spec",
            )

        success = is_affirmative_verdict(record.operator_judgement)
        denom = max(path_m, optimal_m)
        efficiency = 1.0 if denom == 0.0 else optimal_m / denom
        value = efficiency if success else 0.0
        return Score(
            value=value,
            explanation=(
                f"{'success' if success else 'not successful'}; "
                f"path {path_m:.2f} m vs optimal {optimal_m:.2f} m"
            ),
        )


def spl(state_key: str = "gps") -> Scorer:
    """Success weighted by Path Length: success (operator verdict) scaled by
    how efficient the recorded ``state_key`` path was relative to the optimal
    one. ``1.0`` is a perfectly efficient success; ``0.0`` is any failure or
    missing data. See ``_SuccessWeightedByPathLength`` for the exact
    per-episode formula and its degenerate cases.
    """
    return _SuccessWeightedByPathLength(state_key=state_key)


@dataclass(frozen=True)
class _SuccessWeightedByCompletionTime:
    """Success weighted by Completion Time (Yokoyama & Ha 2021): per-episode
    ``S * T / max(C, T)`` — the completion-time analog of SPL. Where SPL can't
    tell a fast-but-roundabout path from a slow-but-direct one apart if both
    cover the same ground distance, SCT judges time efficiency directly, which
    is what actually matters once an agent's speed limits are part of what's
    being evaluated.

    Real-hardware analog: the paper's own ``T`` is "the shortest possible
    amount of time to reach the goal circumventing obstacles based on the
    agent's dynamics," computed there via an RRT*-Unicycle planner. No such
    dynamics-aware planner is available on this stack, so:

    - ``S``: the recorded operator verdict (``is_affirmative_verdict`` of
      ``record.operator_judgement``), same signal ``spl``/``operator_scorer``
      report.
    - ``C``: the trial's actual elapsed time (seconds) between the first and
      last recorded ``state_key`` fix, read from ``Observation.state_time`` —
      a real wall-clock timestamp the embodiment stamps on every observation
      (confirmed populated by the config-driven rosboard embodiment).
    - ``T``: ``Target.spec["optimal_time"]`` (seconds) if given; else
      ``Target.spec["optimal_path_length"]`` (meters, the same key ``spl``
      reads) divided by ``max_velocity_mps``; else straight-line (haversine)
      distance from the first recorded fix to ``Target.spec["goal_lat"]``/
      ``"goal_lon"]``, also divided by ``max_velocity_mps``. Dividing a
      straight-line distance by a constant max velocity can only
      under-estimate the paper's own obstacle-aware ``T`` (a straight line at
      full speed is never slower than any real detour), so this SCT reads
      conservative relative to the textbook metric, same posture as
      ``spl``'s own ``l_i`` fallback.

    Degenerate cases mirror ``spl``: no ``state_key`` ever recorded, or no way
    to determine ``T``, both score ``0.0``. ``C == T == 0`` scores ``1.0`` by
    convention when successful, avoiding a 0/0 divide.
    """

    state_key: str = "gps"
    max_velocity_mps: float = 0.3  # matches _omnivla_model.py's own DEFAULT_MAXV
    name: str = "sct"

    def __post_init__(self) -> None:
        if self.max_velocity_mps <= 0:
            raise ValueError(f"max_velocity_mps must be > 0, got {self.max_velocity_mps!r}")

    def __call__(self, record: TrialRecord, target: Target | None) -> Score:
        first, first_time, last_time, _path_m = _gps_trace(record, self.state_key)
        if first is None:
            return Score(
                value=0.0,
                explanation=f"no {self.state_key!r} state recorded in this trial",
            )

        optimal_time_s: float | None = None
        if target is not None and "optimal_time" in target.spec:
            optimal_time_s = float(target.spec["optimal_time"])
        elif target is not None and "optimal_path_length" in target.spec:
            optimal_time_s = float(target.spec["optimal_path_length"]) / self.max_velocity_mps
        elif target is not None and "goal_lat" in target.spec and "goal_lon" in target.spec:
            optimal_distance_m = _haversine_m(
                first[0], first[1], float(target.spec["goal_lat"]), float(target.spec["goal_lon"])
            )
            optimal_time_s = optimal_distance_m / self.max_velocity_mps
        if optimal_time_s is None:
            return Score(
                value=0.0,
                explanation=(
                    "no optimal_time, optimal_path_length, or goal_lat/goal_lon in Target.spec"
                ),
            )

        success = is_affirmative_verdict(record.operator_judgement)
        completion_time_s = last_time - first_time
        denom = max(completion_time_s, optimal_time_s)
        efficiency = 1.0 if denom == 0.0 else optimal_time_s / denom
        value = efficiency if success else 0.0
        return Score(
            value=value,
            explanation=(
                f"{'success' if success else 'not successful'}; "
                f"completion {completion_time_s:.2f}s vs optimal {optimal_time_s:.2f}s"
            ),
        )


def sct(state_key: str = "gps", max_velocity_mps: float = 0.3) -> Scorer:
    """Success weighted by Completion Time: success (operator verdict) scaled
    by how efficient the recorded ``state_key`` trace was in time relative to
    an estimated optimal completion time. ``1.0`` is a perfectly efficient
    success; ``0.0`` is any failure or missing data. Raises ``ValueError`` if
    ``max_velocity_mps`` isn't positive. See
    ``_SuccessWeightedByCompletionTime`` for the exact per-episode formula and
    its degenerate cases.
    """
    return _SuccessWeightedByCompletionTime(state_key=state_key, max_velocity_mps=max_velocity_mps)


class VLMScorer:
    """Reserved interface (R10): score from a VLM classifier over final frames.

    VLM judging shipped as a grader instead (R6: scorers must stay pure
    readers of the record); instantiating and calling this raises so the
    reserved contract stays visible without half-baked behavior.
    """

    name = "vlm"

    def __call__(self, record: TrialRecord, target: Target | None) -> Score:
        """Fail explicitly because VLM judging ships as the 'vlm' grader instead."""
        raise NotImplementedError(
            "VLM judging ships as the 'vlm' grader (--grader vlm) with the "
            "'operator' scorer reading its judgement"
        )
