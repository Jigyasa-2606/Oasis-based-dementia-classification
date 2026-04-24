"""Extract OASIS subject ID from slice filenames (e.g. OAS1_0137_MR1_mpr-3_139.jpg -> OAS1_0137)."""

from __future__ import annotations

import re
from pathlib import Path

_OASIS_SUBJECT_RE = re.compile(r"^(OAS1_\d+)_", re.IGNORECASE)


def extract_subject_id(path: str | Path) -> str:
    name = Path(path).name
    m = _OASIS_SUBJECT_RE.match(name)
    if m:
        return m.group(1).upper()
    stem = Path(name).stem
    return stem
