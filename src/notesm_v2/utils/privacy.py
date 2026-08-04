from __future__ import annotations

import hashlib


def stable_hash(value: object, salt: str = "notesm-v2") -> str:
    payload = f"{salt}:{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
