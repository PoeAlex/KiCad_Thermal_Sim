"""
Unit tests for pwl_parser module.

Tests PWL file parsing, interpolation, and validation for
time-varying power profiles.
"""

import os
import pytest
import numpy as np

from ThermalSim.pwl_parser import parse_pwl_file, interpolate_pwl, validate_pwl


class TestParsePWLFile:
    """Tests for parse_pwl_file function."""

    def test_parse_basic_pwl(self, tmp_path):
        """Test parsing a basic two-column PWL file with comments."""
        pwl_file = tmp_path / "basic.pwl"
        pwl_file.write_text(
            "; Time(s)  Power(W)\n"
            "* This is also a comment\n"
            "\n"
            "0.0    0.0\n"
            "0.001  1.0\n"
            "0.005  2.5\n"
            "0.010  2.5\n"
            "0.020  0.0\n"
        )
        times, powers = parse_pwl_file(str(pwl_file))

        assert len(times) == 5
        assert len(powers) == 5
        np.testing.assert_array_almost_equal(
            times, [0.0, 0.001, 0.005, 0.010, 0.020]
        )
        np.testing.assert_array_almost_equal(
            powers, [0.0, 1.0, 2.5, 2.5, 0.0]
        )

    def test_parse_tabs_and_spaces(self, tmp_path):
        """Test parsing with mixed whitespace separators."""
        pwl_file = tmp_path / "mixed_ws.pwl"
        pwl_file.write_text(
            "0.0\t0.0\n"
            "1.0  \t  5.0\n"
            "2.0\t\t10.0\n"
        )
        times, powers = parse_pwl_file(str(pwl_file))

        assert len(times) == 3
        np.testing.assert_array_almost_equal(times, [0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(powers, [0.0, 5.0, 10.0])

    def test_parse_empty_file(self, tmp_path):
        """Test that an empty file raises ValueError."""
        pwl_file = tmp_path / "empty.pwl"
        pwl_file.write_text("; only comments\n* nothing here\n\n")

        with pytest.raises(ValueError, match="no data points"):
            parse_pwl_file(str(pwl_file))

    def test_parse_non_monotonic(self, tmp_path):
        """Test that non-monotonic times raise ValueError."""
        pwl_file = tmp_path / "non_mono.pwl"
        pwl_file.write_text(
            "0.0  0.0\n"
            "1.0  1.0\n"
            "0.5  2.0\n"  # Goes backward
        )

        with pytest.raises(ValueError, match="not strictly increasing"):
            parse_pwl_file(str(pwl_file))

    def test_parse_duplicate_times(self, tmp_path):
        """Test that duplicate times raise ValueError."""
        pwl_file = tmp_path / "dup_times.pwl"
        pwl_file.write_text(
            "0.0  0.0\n"
            "1.0  1.0\n"
            "1.0  2.0\n"  # Duplicate time
        )

        with pytest.raises(ValueError, match="not strictly increasing"):
            parse_pwl_file(str(pwl_file))

    def test_parse_single_point(self, tmp_path):
        """Test that a single data row is valid."""
        pwl_file = tmp_path / "single.pwl"
        pwl_file.write_text("0.0  1.5\n")

        times, powers = parse_pwl_file(str(pwl_file))
        assert len(times) == 1
        assert times[0] == 0.0
        assert powers[0] == 1.5

    def test_parse_file_not_found(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_pwl_file("/nonexistent/path/file.pwl")

    def test_parse_bad_format(self, tmp_path):
        """Test that lines with wrong number of columns raise ValueError."""
        pwl_file = tmp_path / "bad.pwl"
        pwl_file.write_text("0.0\n")  # Only one column

        with pytest.raises(ValueError, match="expected 2 columns"):
            parse_pwl_file(str(pwl_file))

    def test_parse_non_numeric(self, tmp_path):
        """Test that non-numeric values raise ValueError."""
        pwl_file = tmp_path / "nonnumeric.pwl"
        pwl_file.write_text("abc  def\n")

        with pytest.raises(ValueError, match="cannot parse"):
            parse_pwl_file(str(pwl_file))

    def test_parse_extra_columns_ignored(self, tmp_path):
        """Test that extra columns beyond 2 are tolerated."""
        pwl_file = tmp_path / "extra_cols.pwl"
        pwl_file.write_text(
            "0.0  0.0  extra\n"
            "1.0  5.0  stuff\n"
        )
        # Should parse first two columns without error
        times, powers = parse_pwl_file(str(pwl_file))
        assert len(times) == 2


class TestInterpolatePWL:
    """Tests for interpolate_pwl function."""

    @pytest.fixture
    def ramp_profile(self):
        """Simple ramp: 0W at t=0, 10W at t=1."""
        return np.array([0.0, 1.0]), np.array([0.0, 10.0])

    def test_interpolation_midpoint(self, ramp_profile):
        """Test linear interpolation between breakpoints."""
        times, powers = ramp_profile
        result = interpolate_pwl(times, powers, 0.5)
        assert abs(result - 5.0) < 1e-10

    def test_interpolation_exact_breakpoint(self, ramp_profile):
        """Test exact match at a breakpoint."""
        times, powers = ramp_profile
        assert abs(interpolate_pwl(times, powers, 0.0) - 0.0) < 1e-10
        assert abs(interpolate_pwl(times, powers, 1.0) - 10.0) < 1e-10

    def test_interpolation_clamp_before(self, ramp_profile):
        """Test that values before t_min clamp to first value."""
        times, powers = ramp_profile
        result = interpolate_pwl(times, powers, -1.0)
        assert abs(result - 0.0) < 1e-10

    def test_interpolation_clamp_after(self, ramp_profile):
        """Test that values after t_max clamp to last value."""
        times, powers = ramp_profile
        result = interpolate_pwl(times, powers, 5.0)
        assert abs(result - 10.0) < 1e-10

    def test_interpolation_multi_segment(self):
        """Test interpolation across multiple segments."""
        times = np.array([0.0, 1.0, 2.0, 3.0])
        powers = np.array([0.0, 10.0, 5.0, 15.0])

        # Midpoint of first segment
        assert abs(interpolate_pwl(times, powers, 0.5) - 5.0) < 1e-10
        # Midpoint of second segment (decreasing)
        assert abs(interpolate_pwl(times, powers, 1.5) - 7.5) < 1e-10
        # Midpoint of third segment
        assert abs(interpolate_pwl(times, powers, 2.5) - 10.0) < 1e-10

    def test_interpolation_single_point(self):
        """Test interpolation with single data point returns that value."""
        times = np.array([1.0])
        powers = np.array([5.0])

        assert abs(interpolate_pwl(times, powers, 0.0) - 5.0) < 1e-10
        assert abs(interpolate_pwl(times, powers, 1.0) - 5.0) < 1e-10
        assert abs(interpolate_pwl(times, powers, 99.0) - 5.0) < 1e-10


class TestValidatePWL:
    """Tests for validate_pwl function."""

    def test_validate_negative_power(self):
        """Test that negative power values produce a warning."""
        times = np.array([0.0, 1.0, 2.0])
        powers = np.array([1.0, -0.5, 1.0])

        warnings = validate_pwl(times, powers)
        assert any("Negative" in w for w in warnings)

    def test_validate_no_warnings(self):
        """Test that a clean profile produces no warnings."""
        times = np.array([0.0, 1.0, 2.0])
        powers = np.array([0.0, 5.0, 10.0])

        warnings = validate_pwl(times, powers)
        assert len(warnings) == 0

    def test_validate_very_short_segment(self):
        """Test warning for very short time segments."""
        times = np.array([0.0, 1e-10, 1.0])
        powers = np.array([0.0, 5.0, 10.0])

        warnings = validate_pwl(times, powers)
        assert any("short" in w.lower() for w in warnings)

    def test_validate_very_high_power(self):
        """Test warning for very high power values."""
        times = np.array([0.0, 1.0])
        powers = np.array([0.0, 5000.0])

        warnings = validate_pwl(times, powers)
        assert any("high" in w.lower() for w in warnings)
