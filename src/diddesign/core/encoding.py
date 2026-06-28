"""Automatic string-to-integer encoding for DID structural variables.

Mathematical justification:
- time column: Correct temporal ordering is critical for sDID (ΔY_t = Y_t - Y_{t-1}).
  Strings are encoded by lexicographic order (matching Stata's egen group() behavior).
- unit_id column: Only grouping matters (ordering irrelevant for bootstrap clustering).
- cluster column: Only grouping matters (ordering irrelevant for bootstrap blocking).
"""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd

from ..errors import WarningCode, did_warn


def auto_encode_string_columns(
    df: pd.DataFrame,
    *,
    time: str | None = None,
    unit_id: str | None = None,
    id_cluster: str | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[Any, int]]]:
    """Automatically encode string columns to integers for DID estimation.

    Parameters
    ----------
    df : pd.DataFrame
        Input data frame.
    time : str
        Time column name.
    unit_id : str | None
        Unit identifier column name.
    id_cluster : str | None
        Cluster variable column name.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, dict[Any, int]]]
        Encoded DataFrame and encoding maps {column_name: {original_value: encoded_int}}.

    Notes
    -----
    Encoding strategy (matches Stata's ``egen group()`` behavior):
    - time: Sorted lexicographically then assigned 0, 1, 2, ...
    - unit_id: Sorted lexicographically then assigned 0, 1, 2, ...
    - id_cluster: Sorted lexicographically then assigned 0, 1, 2, ...

    All columns use sorted order to ensure deterministic, reproducible encoding
    that matches ``pd.factorize(sort=True)`` and Stata ``egen group()``.

    Issues W001 warning for each auto-encoded column.
    Does NOT encode columns that are already numeric.
    """
    encoded_df = df.copy()
    encoding_maps: dict[str, dict[Any, int]] = {}

    columns_to_check = []
    if time is not None:
        columns_to_check.append((time, "time"))
    if unit_id is not None:
        columns_to_check.append((unit_id, "unit_id"))
    if id_cluster is not None and id_cluster != unit_id:
        columns_to_check.append((id_cluster, "id_cluster"))

    for col_name, role in columns_to_check:
        if col_name is None or col_name not in df.columns:
            continue

        # Check if column contains string values
        col_values = df[col_name]
        if not _is_string_column(col_values):
            continue

        # Build encoding map: always sorted (matches Stata's egen group() behavior)
        unique_values = sorted(col_values.dropna().unique())

        encoding_map = {val: idx for idx, val in enumerate(unique_values)}
        encoding_maps[col_name] = encoding_map

        # Apply encoding
        encoded_df[col_name] = col_values.map(encoding_map)

        # Emit warning
        did_warn(
            WarningCode.W001,
            f"String column '{col_name}' ({role}) automatically encoded to integer.",
            context={"column": col_name, "role": role, "n_levels": len(encoding_map)},
            stacklevel=3,
        )

    return encoded_df, encoding_maps


def _is_string_column(series: pd.Series) -> bool:
    """Check if a pandas Series contains string (object/string) dtype."""
    if series.dtype == object:
        non_null = series.dropna()
        if len(non_null) == 0:
            return False
        return all(isinstance(v, str) for v in non_null.head(100))
    return pd.api.types.is_string_dtype(series)
