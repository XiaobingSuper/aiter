# SPDX-License-Identifier: MIT
"""Auto-install the shape collector in every process that inherits PYTHONPATH.

Putting this directory on PYTHONPATH makes each ATOM worker (TP/EP ranks are
spawned children) install the hooks on its own. Without ATOM_SHAPE_DUMP set it
does nothing, so leaving the path exported is harmless.

Any sitecustomize further down sys.path still runs — this one chains to it.
"""

import os
import sys
from importlib.machinery import PathFinder

_here = os.path.dirname(os.path.abspath(__file__))

_rest = [p for p in sys.path if os.path.abspath(p or os.curdir) != _here]
_spec = PathFinder.find_spec("sitecustomize", _rest)
if _spec is not None and _spec.loader is not None:
    try:
        _spec.loader.exec_module(sys.modules[__name__])
    except Exception:
        pass

if os.environ.get("ATOM_SHAPE_DUMP"):
    try:
        import collector

        collector.install()
    except Exception as exc:  # never break the interpreter over collection
        print(f"[model_shapes] collector not installed: {exc}", file=sys.stderr)
