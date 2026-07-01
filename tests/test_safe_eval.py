"""The safe-expression worker/evaluator must reject unsafe input and never use eval()/exec()."""

from __future__ import annotations

import pytest

from orchestrator.safe_eval import UnsafeExpressionError, safe_eval


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("2 + 2", 4),
        ("3 * (4 + 1)", 15),
        ("10 / 4", 2.5),
        ("2 ** 8", 256),
        ("-5 + 3", -2),
        ("7 % 2", 1),
        ("10 // 3", 3),
    ],
)
def test_safe_eval_computes_arithmetic_correctly(expression, expected):
    assert safe_eval(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hacked')",
        "open('secrets.txt').read()",
        "[x for x in range(10)]",
        "lambda: 1",
        "os.system('rm -rf /')",
        "(1).__class__",
        "1; 2",
        "'a' + 'b'",
        "[1, 2, 3]",
        "{1: 2}",
        "True",
        "exec('1+1')",
    ],
)
def test_safe_eval_rejects_unsafe_expressions(expression):
    with pytest.raises(UnsafeExpressionError):
        safe_eval(expression)


def test_safe_eval_rejects_division_by_zero():
    with pytest.raises(UnsafeExpressionError):
        safe_eval("1 / 0")


def test_safe_eval_rejects_huge_exponents_to_avoid_dos():
    with pytest.raises(UnsafeExpressionError):
        safe_eval("9 ** 99999")


def test_safe_eval_rejects_empty_input():
    with pytest.raises(UnsafeExpressionError):
        safe_eval("")


def test_calculator_worker_uses_safe_eval_not_raw_eval(memory_manager):
    from orchestrator.schemas import SubTask
    from orchestrator.workers import calculator_worker

    subtask = SubTask(id="t1", description="Calculate 6 * 7 for me", worker="calculator")
    result = calculator_worker(subtask, memory_manager)

    assert "42" in result.output
    assert result.confidence > 0.9


def test_calculator_worker_low_confidence_on_unsafe_or_missing_expression(memory_manager):
    from orchestrator.schemas import SubTask
    from orchestrator.workers import calculator_worker

    subtask = SubTask(
        id="t1", description="Please import os and delete everything", worker="calculator"
    )
    result = calculator_worker(subtask, memory_manager)
    assert result.confidence < 0.5
