"""C2PA / camera-attestation short-circuit scaffold (IMPROVEMENTS.md #24).

If a file carries a valid C2PA manifest signed by a known device or vendor,
we can short-circuit the deepfake pipeline (REAL with high confidence) before
ever loading the model — provenance is a stronger signal than any classifier.
A real implementation will:

* Parse the manifest via the ``c2pa`` Python bindings (or shell out to ``c2patool``).
* Verify the signature chain against a configurable trust list.
* Return a ``ProvenanceVerdict`` with the signer, claim generator, and a
  ``trust_level`` enum (verified / unsigned / tampered).

Today this module exposes the shape so callers can stub out integration.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class TrustLevel(str, Enum):
    UNSIGNED = "unsigned"
    VERIFIED = "verified"
    TAMPERED = "tampered"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProvenanceVerdict:
    trust_level: TrustLevel
    signer: str | None = None
    claim_generator: str | None = None
    details: str | None = None


def inspect(path: Path) -> ProvenanceVerdict:  # pragma: no cover - scaffold
    """Return a ProvenanceVerdict for ``path``. NOT YET WIRED.

    The default until a real C2PA reader is integrated is ``UNSIGNED`` so the
    pipeline behaves identically to today (no short-circuit).
    """
    return ProvenanceVerdict(trust_level=TrustLevel.UNSIGNED)
