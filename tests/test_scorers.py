"""Scorer builtins, epoch reducers, and the operator-event scorer."""

from __future__ import annotations

import numpy as np
import pytest

from inspect_robots.rollout import StepRecord, TrialRecord
from inspect_robots.scene import Target
from inspect_robots.scorer import (
    Score,
    VLMScorer,
    distance_to_goal,
    episode_length,
    get_reducer,
    is_affirmative_verdict,
    min_distance_to_goal,
    operator_scorer,
    reached_goal_state,
    reduce_scores,
    sct,
    spl,
    success_at_end,
)
from inspect_robots.types import Action, Observation, StepResult


def _record(distances: list[float], *, success: bool, operator: str | None = None) -> TrialRecord:
    steps = []
    for t, d in enumerate(distances):
        last = t == len(distances) - 1
        steps.append(
            StepRecord(
                t=t,
                observation=Observation(),
                action=Action(data=np.zeros(2)),
                result=StepResult(
                    observation=Observation(),
                    terminated=last and success,
                    termination_reason="success" if (last and success) else None,
                    info={"distance": d},
                ),
            )
        )
    rec = TrialRecord(scene_id="s", epoch=0, seed=0, steps=steps)
    rec.terminated = success
    rec.termination_reason = "success" if success else None
    rec.operator_judgement = operator
    return rec


def test_success_at_end() -> None:
    assert success_at_end()(_record([0.5, 0.0], success=True), None).value is True
    assert success_at_end()(_record([0.5, 0.3], success=False), None).value is False


def test_episode_length() -> None:
    assert episode_length()(_record([1.0, 0.5, 0.0], success=True), None).value == 3


def test_min_distance_to_goal() -> None:
    assert min_distance_to_goal()(_record([0.9, 0.2, 0.4], success=False), None).value == 0.2


def test_reached_goal_state() -> None:
    assert reached_goal_state(0.05)(_record([0.5, 0.02], success=True), None).value is True
    assert reached_goal_state(0.05)(_record([0.5, 0.2], success=False), None).value is False


def _gps_record(
    states: list[np.ndarray | None],
    *,
    key: str = "gps",
    operator: str | None = None,
    times: list[float] | None = None,
) -> TrialRecord:
    times = times if times is not None else [0.0] * len(states)
    steps = [
        StepRecord(
            t=t,
            observation=Observation(state={key: s} if s is not None else {}, state_time=times[t]),
            action=Action(data=np.zeros(2)),
            result=StepResult(observation=Observation(), terminated=False, termination_reason=None),
        )
        for t, s in enumerate(states)
    ]
    rec = TrialRecord(scene_id="s", epoch=0, seed=0, steps=steps)
    rec.operator_judgement = operator
    return rec


def test_distance_to_goal_no_target_is_infinite() -> None:
    record = _gps_record([np.array([0.0, 0.0])])
    score = distance_to_goal()(record, None)
    assert score.value == float("inf")
    assert "no goal" in (score.explanation or "")


def test_distance_to_goal_target_missing_goal_fields_is_infinite() -> None:
    record = _gps_record([np.array([0.0, 0.0])])
    score = distance_to_goal()(record, Target(kind="reach_goal", spec={}))
    assert score.value == float("inf")


def test_distance_to_goal_no_gps_recorded_is_infinite() -> None:
    record = _gps_record([None, None])
    target = Target(kind="reach_goal", spec={"goal_lat": 1.0, "goal_lon": 1.0})
    score = distance_to_goal()(record, target)
    assert score.value == float("inf")
    assert "no 'gps' state recorded" in (score.explanation or "")


def test_distance_to_goal_zero_at_the_goal() -> None:
    record = _gps_record([np.array([10.0, 20.0])])
    target = Target(kind="reach_goal", spec={"goal_lat": 10.0, "goal_lon": 20.0})
    assert distance_to_goal()(record, target).value == pytest.approx(0.0, abs=1e-6)


def test_distance_to_goal_known_reference_distance() -> None:
    # 1 degree of longitude at the equator: circumference / 360, matching the
    # implementation's own mean-Earth-radius constant (not an independent source,
    # since haversine has no simpler closed form to check against by hand) --
    # this pins the constant and the formula together against regression.
    record = _gps_record([np.array([0.0, 0.0])])
    target = Target(kind="reach_goal", spec={"goal_lat": 0.0, "goal_lon": 1.0})
    expected = 2 * 3.14159265358979 * 6371008.8 / 360
    assert distance_to_goal()(record, target).value == pytest.approx(expected, rel=1e-6)


def test_distance_to_goal_uses_last_recorded_gps_not_first() -> None:
    far = np.array([45.0, 45.0])
    at_goal = np.array([1.0, 1.0])
    record = _gps_record([far, at_goal])
    target = Target(kind="reach_goal", spec={"goal_lat": 1.0, "goal_lon": 1.0})
    assert distance_to_goal()(record, target).value == pytest.approx(0.0, abs=1e-6)


def test_distance_to_goal_skips_steps_missing_state_to_find_the_last_present() -> None:
    at_goal = np.array([1.0, 1.0])
    record = _gps_record([at_goal, None])  # last step recorded no gps at all
    target = Target(kind="reach_goal", spec={"goal_lat": 1.0, "goal_lon": 1.0})
    assert distance_to_goal()(record, target).value == pytest.approx(0.0, abs=1e-6)


def test_distance_to_goal_respects_custom_state_key() -> None:
    record = _gps_record([np.array([1.0, 1.0])], key="fix")
    target = Target(kind="reach_goal", spec={"goal_lat": 1.0, "goal_lon": 1.0})
    assert distance_to_goal(state_key="gps")(record, target).value == float("inf")
    assert distance_to_goal(state_key="fix")(record, target).value == pytest.approx(0.0, abs=1e-6)


def test_spl_no_gps_at_all_is_zero() -> None:
    record = _gps_record([None, None], operator="success")
    target = Target(kind="reach_goal", spec={"goal_lat": 1.0, "goal_lon": 1.0})
    score = spl()(record, target)
    assert score.value == 0.0
    assert "no 'gps' state recorded" in (score.explanation or "")


def test_spl_no_target_is_zero() -> None:
    record = _gps_record([np.array([0.0, 0.0])], operator="success")
    assert spl()(record, None).value == 0.0


def test_spl_target_missing_optimal_and_goal_fields_is_zero() -> None:
    record = _gps_record([np.array([0.0, 0.0])], operator="success")
    score = spl()(record, Target(kind="reach_goal", spec={}))
    assert score.value == 0.0
    assert "no optimal_path_length or goal_lat/goal_lon" in (score.explanation or "")


def test_spl_zero_optimal_path_and_already_at_goal_is_one_by_convention() -> None:
    # Single fix -> path_m == 0.0; optimal_path_length == 0.0 too -> 0/0 guarded.
    record = _gps_record([np.array([1.0, 1.0])], operator="success")
    target = Target(kind="reach_goal", spec={"optimal_path_length": 0.0})
    assert spl()(record, target).value == pytest.approx(1.0)


def test_spl_perfect_efficiency_is_one() -> None:
    # Second fix *is* the goal, so the fallback optimal length equals the
    # one traveled segment exactly.
    start = np.array([0.0, 0.0])
    goal_point = np.array([0.0, 1.0])
    record = _gps_record([start, goal_point], operator="success")
    target = Target(kind="reach_goal", spec={"goal_lat": 0.0, "goal_lon": 1.0})
    assert spl()(record, target).value == pytest.approx(1.0)


def test_spl_success_but_took_2x_optimal_path_is_half() -> None:
    # Along the equator: 0 -> 1.5 -> 1 (lon degrees), goal at lon 1. Traveled
    # path is 1.5 + 0.5 = 2.0 degrees-worth; optimal (straight line from the
    # first fix) is 1.0 degree-worth -> exactly half, no float slop, since
    # haversine reduces to a linear function of the longitude delta at lat=0.
    record = _gps_record(
        [np.array([0.0, 0.0]), np.array([0.0, 1.5]), np.array([0.0, 1.0])],
        operator="success",
    )
    target = Target(kind="reach_goal", spec={"goal_lat": 0.0, "goal_lon": 1.0})
    assert spl()(record, target).value == pytest.approx(0.5, rel=1e-9)


def test_spl_failure_is_zero_regardless_of_path_efficiency() -> None:
    # Same perfectly-efficient path as the success case, but no affirmative verdict.
    start = np.array([0.0, 0.0])
    goal_point = np.array([0.0, 1.0])
    target = Target(kind="reach_goal", spec={"goal_lat": 0.0, "goal_lon": 1.0})
    record_fail = _gps_record([start, goal_point], operator="fail")
    record_none = _gps_record([start, goal_point], operator=None)
    assert spl()(record_fail, target).value == 0.0
    assert spl()(record_none, target).value == 0.0


def test_spl_prefers_explicit_optimal_path_length_over_goal_fallback() -> None:
    # path_m from (0,0)->(0,1) is ~111_194.9 m (1 degree of longitude at the
    # equator); an explicit optimal_path_length of 500 m must win over the
    # goal-fallback computation, which would also equal ~111_194.9 m here.
    record = _gps_record([np.array([0.0, 0.0]), np.array([0.0, 1.0])], operator="success")
    target = Target(
        kind="reach_goal",
        spec={"optimal_path_length": 500.0, "goal_lat": 0.0, "goal_lon": 1.0},
    )
    path_m = 2 * 3.14159265358979 * 6371008.8 / 360
    assert spl()(record, target).value == pytest.approx(500.0 / path_m, rel=1e-6)


def test_spl_respects_custom_state_key() -> None:
    record = _gps_record(
        [np.array([0.0, 0.0]), np.array([0.0, 1.0])], key="fix", operator="success"
    )
    target = Target(kind="reach_goal", spec={"goal_lat": 0.0, "goal_lon": 1.0})
    assert spl(state_key="gps")(record, target).value == 0.0
    assert spl(state_key="fix")(record, target).value == pytest.approx(1.0)


def test_spl_name_is_spl() -> None:
    assert spl().name == "spl"


def test_sct_no_gps_at_all_is_zero() -> None:
    record = _gps_record([None, None], operator="success")
    target = Target(kind="reach_goal", spec={"goal_lat": 1.0, "goal_lon": 1.0})
    score = sct()(record, target)
    assert score.value == 0.0
    assert "no 'gps' state recorded" in (score.explanation or "")


def test_sct_no_target_is_zero() -> None:
    record = _gps_record([np.array([0.0, 0.0])], operator="success")
    assert sct()(record, None).value == 0.0


def test_sct_target_missing_all_time_fields_is_zero() -> None:
    record = _gps_record([np.array([0.0, 0.0])], operator="success")
    score = sct()(record, Target(kind="reach_goal", spec={}))
    assert score.value == 0.0
    assert "no optimal_time, optimal_path_length, or goal_lat/goal_lon" in (score.explanation or "")


def test_sct_prefers_explicit_optimal_time_over_everything_else() -> None:
    # optimal_path_length and goal_lat/goal_lon are both also present and both
    # disagree with optimal_time -- the explicit value must win.
    record = _gps_record(
        [np.array([0.0, 0.0]), np.array([0.0, 0.0])],
        operator="success",
        times=[0.0, 1000.0],
    )
    target = Target(
        kind="reach_goal",
        spec={
            "optimal_time": 500.0,
            "optimal_path_length": 1.0,
            "goal_lat": 0.0,
            "goal_lon": 0.0,
        },
    )
    assert sct()(record, target).value == pytest.approx(0.5)


def test_sct_falls_back_to_optimal_path_length_over_velocity() -> None:
    # 30 m at the default 0.3 m/s max velocity -> 100 s optimal; completion
    # matches exactly -> perfect efficiency.
    record = _gps_record(
        [np.array([0.0, 0.0]), np.array([0.0, 0.0])],
        operator="success",
        times=[0.0, 100.0],
    )
    target = Target(kind="reach_goal", spec={"optimal_path_length": 30.0})
    assert sct()(record, target).value == pytest.approx(1.0)


def test_sct_falls_back_to_goal_haversine_over_velocity() -> None:
    # 1 degree of longitude at the equator / 0.3 m/s max velocity is the
    # optimal time; taking exactly twice that long -> 0.5.
    optimal_time_s = (2 * 3.14159265358979 * 6371008.8 / 360) / 0.3
    record = _gps_record(
        [np.array([0.0, 0.0]), np.array([0.0, 0.0])],
        operator="success",
        times=[0.0, 2 * optimal_time_s],
    )
    target = Target(kind="reach_goal", spec={"goal_lat": 0.0, "goal_lon": 1.0})
    assert sct()(record, target).value == pytest.approx(0.5, rel=1e-9)


def test_sct_zero_completion_and_zero_optimal_is_one_by_convention() -> None:
    record = _gps_record([np.array([1.0, 1.0])], operator="success")
    target = Target(kind="reach_goal", spec={"optimal_time": 0.0})
    assert sct()(record, target).value == pytest.approx(1.0)


def test_sct_failure_is_zero_regardless_of_time_efficiency() -> None:
    record = _gps_record(
        [np.array([0.0, 0.0]), np.array([0.0, 0.0])],
        operator="fail",
        times=[0.0, 100.0],
    )
    target = Target(kind="reach_goal", spec={"optimal_path_length": 30.0})
    assert sct()(record, target).value == 0.0


def test_sct_respects_custom_state_key() -> None:
    record = _gps_record(
        [np.array([0.0, 0.0]), np.array([0.0, 0.0])],
        key="fix",
        operator="success",
        times=[0.0, 100.0],
    )
    target = Target(kind="reach_goal", spec={"optimal_path_length": 30.0})
    assert sct(state_key="gps")(record, target).value == 0.0
    assert sct(state_key="fix")(record, target).value == pytest.approx(1.0)


def test_sct_rejects_nonpositive_max_velocity() -> None:
    with pytest.raises(ValueError, match="max_velocity_mps"):
        sct(max_velocity_mps=0.0)


def test_sct_name_is_sct() -> None:
    assert sct().name == "sct"


def test_operator_scorer_reads_recorded_verdict() -> None:
    assert operator_scorer()(_record([0.5], success=False, operator="success"), None).value is True
    assert operator_scorer()(_record([0.5], success=False, operator="fail"), None).value is False
    # No verdict recorded (unattended run): defaults to not-successful.
    assert operator_scorer()(_record([0.5], success=False), None).value is False


@pytest.mark.parametrize("verdict", ["success", "pass", "yes", "y", "1", "true"])
def test_is_affirmative_verdict_accepts_the_recognized_vocabulary(verdict: str) -> None:
    assert is_affirmative_verdict(verdict) is True


@pytest.mark.parametrize("verdict", ["YES", "Success", "  y  ", "\tPASS\n"])
def test_is_affirmative_verdict_ignores_case_and_surrounding_whitespace(verdict: str) -> None:
    # Operators type free-form text; the comparison rules are part of the
    # contract, not incidental to it.
    assert is_affirmative_verdict(verdict) is True


@pytest.mark.parametrize("verdict", ["fail", "no", "n", "0", "partial", "", "   ", "yes please"])
def test_is_affirmative_verdict_rejects_everything_else(verdict: str) -> None:
    assert is_affirmative_verdict(verdict) is False


def test_is_affirmative_verdict_treats_no_judgement_as_not_affirmative() -> None:
    # Unattended runs record no verdict at all; absence is not assent.
    assert is_affirmative_verdict(None) is False


def test_operator_scorer_agrees_with_the_public_predicate() -> None:
    # The scorer must not carry a second copy of the contract.
    for verdict in ("  Yes ", "fail"):
        score = operator_scorer()(_record([0.5], success=False, operator=verdict), None)
        assert score.value is is_affirmative_verdict(verdict)


def test_reducers_numeric() -> None:
    scores = [Score(value=True), Score(value=False), Score(value=True), Score(value=True)]
    assert reduce_scores("mean", scores).value == 0.75
    assert reduce_scores("max", scores).value == 1.0
    assert reduce_scores("min", scores).value == 0.0


def test_reducer_mode_categorical() -> None:
    scores = [Score(value="a"), Score(value="b"), Score(value="a")]
    assert reduce_scores("mode", scores).value == "a"


def test_mean_over_nonnumeric_string_raises() -> None:
    scores = [Score(value="left"), Score(value="right")]
    with pytest.raises(TypeError, match="non-numeric"):
        reduce_scores("mean", scores)


def test_pass_at_k() -> None:
    # 4 epochs, 1 success: pass@1 = 1/4, pass@4 = 1.0
    scores = [Score(value=True), Score(value=False), Score(value=False), Score(value=False)]
    assert reduce_scores("pass_at_1", scores).value == pytest.approx(0.25)
    assert reduce_scores("pass_at_4", scores).value == pytest.approx(1.0)


def test_unknown_reducer_raises() -> None:
    with pytest.raises(ValueError, match="unknown epoch reducer"):
        get_reducer("nope")


def test_vlm_scorer_stub_points_at_the_vlm_grader() -> None:
    with pytest.raises(NotImplementedError, match=r"'vlm' grader \(--grader vlm\)"):
        VLMScorer()(_record([], success=False), None)
