"""Pytest configuration for benchmarks.

Benchmarks are independent of the main test suite (tests/).
Run with: python -m pytest benchmarks/ -v
"""

import sys
from pathlib import Path

# Ensure the package source is importable when running from diddesign-py/
src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
