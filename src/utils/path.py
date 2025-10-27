"""
Path utilities to keep project code portable after relocating the repository.

The helpers here are intended to replace scattered ad-hoc logic that relied on
the previous absolute location under ``/Volumes/Expansion``. They allow scripts
and modules to resolve resources (metadata, logs, outputs, etc.) relative to the
current working directory or the project root computed from this file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence
import sys


@lru_cache
def get_project_root() -> Path:
    """
    Return the project root directory.

    This assumes ``src/utils/path.py`` lives under ``<root>/src/utils``.
    The cached result avoids repeated filesystem work whenever we resolve
    project-relative paths.
    """
    return Path(__file__).resolve().parents[2]


def _iter_resolution_bases(base: Path | None) -> Iterable[Path]:
    if base is not None:
        yield Path(base).expanduser().resolve()
    yield Path.cwd().resolve()
    yield get_project_root()


def resolve_path(
    value: str | Path,
    *,
    base: Path | None = None,
    allow_missing: bool = False,
) -> Path:
    """
    Resolve ``value`` relative to ``base``/cwd/project root.

    Parameters
    ----------
    value:
        Either an absolute path (returned unchanged) or a relative path that
        should be interpreted against the candidate bases.
    base:
        Optional first candidate base directory. Useful when a CLI flag
        specifies an input/output directory. If the resulting path exists (or
        ``allow_missing`` is True) it is returned before considering the other
        bases.
    allow_missing:
        When True, return the first resolved candidate even if it does not exist.
        This supports callers that plan to create new files/directories (e.g.
        log files). When False, the first *existing* candidate is returned; if
        none exist we fall back to the first candidate regardless so a sensible
        path is still produced.
    """
    path = Path(value).expanduser()
    if path.is_absolute():
        return path

    fallback: Path | None = None
    for candidate_base in _iter_resolution_bases(base):
        candidate = (candidate_base / path).resolve()
        if fallback is None:
            fallback = candidate
        if allow_missing or candidate.exists():
            return candidate

    return fallback if fallback is not None else path


def ensure_project_root_on_syspath(front: bool = True) -> Sequence[str]:
    """
    Add the project root to ``sys.path`` (front by default).

    Returns the updated sys.path list so callers can inspect/debug if needed.
    """
    root = str(get_project_root())
    if root not in sys.path:
        if front:
            sys.path.insert(0, root)
        else:
            sys.path.append(root)
    return sys.path
