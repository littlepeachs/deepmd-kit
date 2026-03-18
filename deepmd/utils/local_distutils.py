from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def get_repo_distutils() -> ModuleType:
    module_name = "_deepmd_repo_distutils"
    loaded = sys.modules.get(module_name)
    if loaded is not None:
        return loaded

    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "distutils.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load repo distutils module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
