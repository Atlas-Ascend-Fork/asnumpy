"""Regression tests for exception assertion helpers."""

import importlib.util
import unittest
from pathlib import Path

import pytest

_ASSERTIONS_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "asnumpy" / "testing" / "_assertions.py"
)
_SPEC = importlib.util.spec_from_file_location("asnumpy_assertions_under_test", _ASSERTIONS_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load assertion helpers from {_ASSERTIONS_PATH}")
_ASSERTIONS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ASSERTIONS)


def _raise_target_assertion():
    raise AssertionError("target assertion")


class TestDirectExceptionAssertions(unittest.TestCase):
    def test_assert_raises_fails_when_target_returns(self):
        with pytest.raises(AssertionError, match="no exception was raised"):
            _ASSERTIONS.assert_raises(AssertionError, lambda: None)

    def test_assert_raises_regex_does_not_match_its_own_failure(self):
        with pytest.raises(AssertionError, match="no exception was raised"):
            _ASSERTIONS.assert_raises_regex(AssertionError, "no exception was raised", lambda: None)

    def test_assert_raises_accepts_target_assertion(self):
        assert _ASSERTIONS.assert_raises(AssertionError, _raise_target_assertion) is None

    def test_assert_raises_regex_matches_target_assertion(self):
        assert (
            _ASSERTIONS.assert_raises_regex(
                AssertionError, "target assertion", _raise_target_assertion
            )
            is None
        )


if __name__ == "__main__":
    unittest.main()
