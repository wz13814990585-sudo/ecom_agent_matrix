from __future__ import annotations

import pytest
from pydantic import ValidationError

from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.modules.agent_cluster.master.policy import (
    INVALID_DEPENDENCY,
    INVALID_TASK_TYPE,
    PLAN_CYCLE,
    PLAN_TOO_LARGE,
    SELF_DEPENDENCY,
    MasterPlanValidationError,
    validate_master_plan,
)
from ecom_agent_matrix.modules.agent_cluster.master.schemas import MasterPlan, PlanStep


def _plan(steps: list[PlanStep]) -> MasterPlan:
    return MasterPlan(
        decision="execute",
        steps=steps,
        confidence=0.9,
        reason_code="TEST",
        planner_source="test",
    )


def _step(step_id: str, **updates) -> PlanStep:
    values = {"step_id": step_id, "agent": "data_query", "task_type": "order_query"}
    values.update(updates)
    return PlanStep(**values)


def test_valid_dag_is_accepted():
    plan = _plan([_step("order_context"), _step("stock_context", depends_on=["order_context"])])
    assert validate_master_plan(plan) is plan


def test_duplicate_step_id_is_rejected():
    with pytest.raises(MasterPlanValidationError) as exc:
        validate_master_plan(_plan([_step("order_context"), _step("order_context")]))
    assert exc.value.code == "DUPLICATE_STEP_ID"


@pytest.mark.parametrize(
    ("steps", "code"),
    [
        ([_step("order_context", depends_on=["missing_context"])], INVALID_DEPENDENCY),
        ([_step("order_context", depends_on=["order_context"])], SELF_DEPENDENCY),
        (
            [
                _step("order_context", depends_on=["stock_context"]),
                _step("stock_context", depends_on=["order_context"]),
            ],
            PLAN_CYCLE,
        ),
    ],
)
def test_invalid_dependencies_are_rejected(steps, code):
    with pytest.raises(MasterPlanValidationError) as exc:
        validate_master_plan(_plan(steps))
    assert exc.value.code == code


def test_unknown_agent_is_rejected_by_typed_schema():
    with pytest.raises(ValidationError):
        _step("unknown_agent_step", agent="tool_agent")


def test_unknown_task_type_is_rejected():
    with pytest.raises(MasterPlanValidationError) as exc:
        validate_master_plan(_plan([_step("unknown_task", task_type="unknown_task")]))
    assert exc.value.code == INVALID_TASK_TYPE


def test_semantic_step_id_is_required():
    with pytest.raises(ValidationError):
        _step("step_0")


def test_max_steps_is_enforced(monkeypatch):
    monkeypatch.setattr(settings, "MASTER_MAX_PLAN_STEPS", 1)
    with pytest.raises(MasterPlanValidationError) as exc:
        validate_master_plan(_plan([_step("order_context"), _step("stock_context")]))
    assert exc.value.code == PLAN_TOO_LARGE
