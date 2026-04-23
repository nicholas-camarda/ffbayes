"""JSON serialization helpers for public artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import date, datetime
from numbers import Integral, Real
from pathlib import Path
from typing import Any


def to_strict_jsonable(value: Any) -> Any:
    """Return a standards-compliant JSON-compatible representation."""
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Mapping):
        return {str(key): to_strict_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [to_strict_jsonable(item) for item in value]
    if isinstance(value, Path | datetime | date):
        return str(value)
    if hasattr(value, 'tolist'):
        return to_strict_jsonable(value.tolist())
    return str(value)


def dumps_strict_json(value: Any, **kwargs: Any) -> str:
    """Serialize JSON while rejecting JavaScript-only NaN/Infinity tokens."""
    return json.dumps(to_strict_jsonable(value), allow_nan=False, **kwargs)
