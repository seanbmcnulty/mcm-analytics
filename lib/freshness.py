"""Reload ``lib.*`` modules whose code on disk is newer than what is in memory.

Streamlit Cloud pulls a new commit into the running process: a page script
reruns with the new source, but modules already in ``sys.modules`` keep their
old code (16 and 18 Nov 2026, see pages/01 and pages/13). Pages that depend on
a changed lib module call :func:`reload_if_changed` at the top.

A module is stale when its file changed since this process last looked, or
(first look) was written after the process started. ``importlib.reload``
updates modules in place, so names imported by the caller stay valid. Pass
modules in dependency order (a module after the ones it imports): once one
reloads, every module after it reloads too. Keep this file small and stable;
it can't reload itself.
"""

from __future__ import annotations

import importlib
import logging
import os

_log = logging.getLogger("mcm.freshness")


def process_start() -> float | None:
    """Epoch seconds this process started (Linux /proc), else None."""
    try:
        with open("/proc/stat") as fh:
            btime = next(float(ln.split()[1]) for ln in fh if ln.startswith("btime"))
        with open("/proc/self/stat") as fh:
            ticks = float(fh.read().rsplit(")", 1)[1].split()[19])
        return btime + ticks / os.sysconf("SC_CLK_TCK")
    except Exception:
        return None


def reload_if_changed(*modules) -> list[str]:
    """Reload each stale module (see module docstring); return reloaded names."""
    started = process_start()
    reloaded: list[str] = []
    stale = False
    for mod in modules:
        path = getattr(mod, "__file__", None)
        try:
            mtime = os.path.getmtime(path) if path else None
        except OSError:
            mtime = None
        seen = getattr(mod, "_mcm_seen_mtime", None)
        changed = mtime is not None and (
            (seen is not None and mtime != seen)
            or (seen is None and started is not None and mtime > started + 5))
        if stale or changed:
            stale = True
            try:
                importlib.reload(mod)
                reloaded.append(mod.__name__)
            except Exception:
                _log.exception("reload %s failed", getattr(mod, "__name__", mod))
        mod._mcm_seen_mtime = mtime
    return reloaded
