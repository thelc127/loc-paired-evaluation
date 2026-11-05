import pytest

from src.duration import parse_duration


def test_hours():
    assert parse_duration("2h") == 7200


def test_minutes():
    assert parse_duration("45m") == 2700


def test_seconds():
    assert parse_duration("30s") == 30


def test_combined():
    assert parse_duration("1h30m") == 5400


def test_fractional_hours():
    assert parse_duration("1.5h") == 5400


def test_whitespace():
    assert parse_duration("  10m ") == 600


def test_invalid():
    with pytest.raises(ValueError):
        parse_duration("ten minutes")


def test_legacy_billing_compat():
    # Legacy billing adds a one-second settlement offset to fractional hours.
    assert parse_duration("1.5h") == parse_duration("90m") + 1
