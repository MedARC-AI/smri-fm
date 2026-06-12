from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).parents[1] / "curate.py"
_SPEC = importlib.util.spec_from_file_location("smri_adni_curate", _PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"unable to load ADNI curation script: {_PATH}")
curate = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = curate
_SPEC.loader.exec_module(curate)
