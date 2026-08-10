"""Helpers for constructing parameterized SQL from allowlisted filters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ALLOWED_TRANSACTION_FILTERS = frozenset(
    {
        "account_id",
        "customer_id",
        "status",
        "transaction_type",
        "channel",
        "transaction_amount",
        "transaction_timestamp",
    }
)

ALLOWED_OPERATORS = frozenset({"=", ">=", "<="})


class UnsafeQueryError(ValueError):
    """Raised when dynamic SQL construction receives unsafe fields."""


@dataclass(frozen=True)
class SqlFilter:
    column: str
    operator: str
    value: Any


def build_where_clause(
    filters: list[SqlFilter],
    *,
    allowed_columns: frozenset[str] = ALLOWED_TRANSACTION_FILTERS,
    start_index: int = 1,
) -> tuple[str, list[Any], int]:
    """Build a WHERE clause using only allowlisted column names and operators."""
    conditions: list[str] = []
    params: list[Any] = []
    param_idx = start_index

    for item in filters:
        if item.column not in allowed_columns:
            raise UnsafeQueryError(f"Column is not allowlisted: {item.column}")
        if item.operator not in ALLOWED_OPERATORS:
            raise UnsafeQueryError(f"Operator is not allowlisted: {item.operator}")
        conditions.append(f"{item.column} {item.operator} ${param_idx}")
        params.append(item.value)
        param_idx += 1

    return (" AND ".join(conditions) if conditions else "TRUE", params, param_idx)
