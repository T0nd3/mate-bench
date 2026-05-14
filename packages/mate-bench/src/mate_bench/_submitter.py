from __future__ import annotations

import uuid

from .paths import CONFIG_DIR

_ID_FILE = CONFIG_DIR / "submitter_id.txt"


def get_anonymous_id() -> str:
    if _ID_FILE.exists():
        return _ID_FILE.read_text().strip()
    aid = str(uuid.uuid4())
    _ID_FILE.parent.mkdir(parents=True, exist_ok=True)
    _ID_FILE.write_text(aid)
    return aid
