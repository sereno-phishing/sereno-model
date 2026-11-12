"""The single cross-validation fold builder, pinned to the folds produced before the merge."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

import sereno_model
from sereno_model.data.split import grouped_folds, make_folds

LABELS = np.array([0, 1] * 60)
GROUPS = np.array([f"d{index % 23}" for index in range(120)])
# SHA-256 of the fold indices produced by stacking.make_folds, diagnose.make_folds and
# source_bias.grouped_folds at b31d71b (n_splits=5, seed=42); all three agreed.
RANDOM_DIGEST = "1cb359882284e48830999561024ac4dc52baaad9a33a03a8583258ca7a537744"
GROUPED_DIGEST = "4469be9401a591cac0beed55e0a03c85289dfd6a2b048887bc8eed4dbacb4cf4"


def _digest(folds) -> str:
    found = hashlib.sha256()
    for train, val in folds:
        found.update(np.asarray(train, dtype=np.int64).tobytes())
        found.update(b"|")
        found.update(np.asarray(val, dtype=np.int64).tobytes())
        found.update(b"#")
    return found.hexdigest()


@pytest.mark.parametrize(
    ("strategy", "expected"), [("random", RANDOM_DIGEST), ("grouped", GROUPED_DIGEST)]
)
def test_folds_are_byte_identical_to_the_previous_builders(strategy, expected):
    folds = make_folds(LABELS, GROUPS, n_splits=5, seed=42, strategy=strategy)

    assert _digest(folds) == expected


def test_grouped_folds_equal_the_grouped_strategy():
    assert _digest(grouped_folds(LABELS, GROUPS, n_splits=5, seed=42)) == GROUPED_DIGEST


def test_the_fold_index_dtype_is_int64():
    builds = (make_folds(LABELS, GROUPS, 5, 42, "random"), grouped_folds(LABELS, GROUPS, 5, 42))
    for folds in builds:
        for train, val in folds:
            assert train.dtype == np.int64 and val.dtype == np.int64


def test_the_models_reuse_the_single_builder():
    from sereno_model.models import stacking

    assert stacking.make_folds is make_folds
    assert stacking.grouped_folds is grouped_folds


def test_no_other_module_builds_its_own_cross_validation_folds():
    package = Path(sereno_model.__file__).parent
    builders = ("def grouped_folds", "def make_folds")
    offenders = [
        str(path.relative_to(package))
        for path in package.rglob("*.py")
        if path.name != "split.py"
        and any(name in path.read_text(encoding="utf-8") for name in builders)
    ]

    assert offenders == []
