"""Checkout-local import shim for the src-layout diddesign package."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_SRC_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "diddesign"
_SRC_INIT = _SRC_PACKAGE_ROOT / "__init__.py"

if not _SRC_INIT.exists():
    raise ModuleNotFoundError(
        "Could not locate the checkout-local src/diddesign package."
    )

_SPEC = importlib.util.spec_from_file_location(
    __name__,
    _SRC_INIT,
    submodule_search_locations=[str(_SRC_PACKAGE_ROOT)],
)
if _SPEC is None or _SPEC.loader is None:
    raise ModuleNotFoundError("Could not create an import spec for src/diddesign.")

_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[__name__] = _MODULE
_SPEC.loader.exec_module(_MODULE)
