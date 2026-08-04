from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MimicPaths:
    admissions: str
    patients: str
    discharge_notes: str
    transfers: str | None = None
    services: str | None = None
