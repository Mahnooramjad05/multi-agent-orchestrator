"""
Safe arithmetic expression evaluator.

Used by the calculator/code worker to execute simple Python expressions
extracted from LLM output WITHOUT using raw eval()/exec() on untrusted text.

We parse the expression into an AST and walk it ourselves, only permitting a
fixed whitelist of node types (numeric literals, +, -, *, /, //, %, **,
unary +/-, and parentheses via the AST's natural nesting). Anything else --
function calls, attribute access, names/identifiers, imports, comprehensions,
subscripts, string literals, etc. -- raises UnsafeExpressionError instead of
being evaluated.
"""

from __future__ import annotations

import ast
import operator
from typing import Any, Callable

_ALLOWED_BINOPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARYOPS: dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Cap exponents so ** can't be used for a denial-of-service (e.g. 9**9**9).
_MAX_POWER_EXPONENT = 1000
_MAX_NUMERIC_MAGNITUDE = 10**12


class UnsafeExpressionError(ValueError):
    """Raised when an expression contains a disallowed construct."""


def safe_eval(expression: str) -> float:
    """
    Safely evaluate a simple arithmetic expression string.

    Supports: + - * / // % ** unary +/- and parentheses, on int/float
    literals only. Raises UnsafeExpressionError for anything else
    (names, calls, attribute access, subscripts, strings, comprehensions,
    lambdas, imports, etc.) -- never falls back to eval()/exec().
    """
    if not isinstance(expression, str) or not expression.strip():
        raise UnsafeExpressionError("Empty or non-string expression.")
    if len(expression) > 200:
        raise UnsafeExpressionError("Expression too long.")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError(f"Could not parse expression: {exc}") from exc

    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            raise UnsafeExpressionError("Boolean literals are not allowed.")
        if isinstance(node.value, (int, float)):
            if abs(node.value) > _MAX_NUMERIC_MAGNITUDE:
                raise UnsafeExpressionError("Numeric literal too large.")
            return node.value
        raise UnsafeExpressionError(f"Disallowed literal type: {type(node.value).__name__}")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_BINOPS:
            raise UnsafeExpressionError(f"Disallowed operator: {op_type.__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if op_type is ast.Pow and (abs(right) > _MAX_POWER_EXPONENT):
            raise UnsafeExpressionError("Exponent too large.")
        if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise UnsafeExpressionError("Division by zero.")
        return _ALLOWED_BINOPS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_UNARYOPS:
            raise UnsafeExpressionError(f"Disallowed unary operator: {op_type.__name__}")
        return _ALLOWED_UNARYOPS[op_type](_eval_node(node.operand))

    # Explicitly reject everything else: Name, Call, Attribute, Subscript,
    # List, Dict, Set, Compare, BoolOp, Lambda, comprehensions, Import, etc.
    raise UnsafeExpressionError(f"Disallowed expression element: {type(node).__name__}")
