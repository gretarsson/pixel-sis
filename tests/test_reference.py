"""Optional full-reference regression against the pre-cleanup implementation."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from pixel_sis import load_reference
from pixel_sis._kernels import run_strength


@pytest.mark.reference
def test_original_reference_trajectories():
    location = os.environ.get("PIXEL_SIS_REFERENCE_DATA")
    if not location:
        pytest.skip("Set PIXEL_SIS_REFERENCE_DATA to the prepared reference directory")
    fixture = json.loads((Path(__file__).parent / "reference_expectations.json").read_text())
    root = Path(location)
    for name, expected in fixture["input_sha256"].items():
        digest = hashlib.sha256()
        with (root / name).open("rb") as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                digest.update(block)
        assert digest.hexdigest() == expected, "Changed reference input: " + name
    model = load_reference(root)
    for case in fixture["cases"]:
        counts, attempts, changes = run_strength(
            *model.arguments, np.array(fixture["times"]), *case["rates"], case["seed"], 2000000
        )
        assert hashlib.sha256(counts.tobytes()).hexdigest() == case["counts_sha256"]
        assert attempts == case["attempts"] and changes == case["changes"]
