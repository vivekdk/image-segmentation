from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC_PACKAGE = _ROOT.parent / "src" / "segmentation"

# This repo uses a `src/` layout for the actual package code. When the project
# is installed, packaging tools handle that layout automatically. When a user
# runs `python -m segmentation.train` directly from a fresh checkout, Python
# would normally only see this top-level package directory and fail to discover
# the modules under `src/segmentation`.
#
# Extending `__path__` makes the package behave like a namespace package across
# both locations, which keeps local developer ergonomics simple without changing
# the actual installable layout.
if _SRC_PACKAGE.exists():
    __path__.append(str(_SRC_PACKAGE))
