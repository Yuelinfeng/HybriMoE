"""Shared metadata context for optional HybriMoE trace events."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator


_TRACE_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("hybrimoe_trace_context", default={})


def get_trace_context() -> dict[str, Any]:
    return dict(_TRACE_CONTEXT.get())


@contextmanager
def trace_context(metadata: dict[str, Any] | None) -> Iterator[None]:
    current = get_trace_context()
    if metadata:
        current.update({key: value for key, value in metadata.items() if value is not None})
    token = _TRACE_CONTEXT.set(current)
    try:
        yield
    finally:
        _TRACE_CONTEXT.reset(token)
