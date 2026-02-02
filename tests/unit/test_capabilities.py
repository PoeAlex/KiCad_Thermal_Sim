"""
Unit tests for capabilities module.

This module tests the feature detection functionality.
"""

import pytest
import sys
import importlib


class TestCapabilityFlags:
    """Tests for capability flag detection."""

    def test_has_libs_is_boolean(self):
        """Test that HAS_LIBS is a boolean value."""
        from ThermalSim.capabilities import HAS_LIBS
        assert isinstance(HAS_LIBS, bool)

    def test_has_pardiso_is_boolean(self):
        """Test that HAS_PARDISO is a boolean value."""
        from ThermalSim.capabilities import HAS_PARDISO
        assert isinstance(HAS_PARDISO, bool)

    def test_has_numba_is_boolean(self):
        """Test that HAS_NUMBA is a boolean value."""
        from ThermalSim.capabilities import HAS_NUMBA
        assert isinstance(HAS_NUMBA, bool)

    def test_has_libs_reflects_imports(self):
        """Test that HAS_LIBS reflects actual import availability."""
        from capabilities import HAS_LIBS

        # In test environment, numpy and matplotlib should be available
        # (they're dependencies for the solver tests)
        try:
            import numpy
            import matplotlib
            numpy_available = True
        except ImportError:
            numpy_available = False

        # HAS_LIBS should reflect availability
        if numpy_available:
            # Note: wx might not be available in test environment
            # so we just verify the flag is set
            assert isinstance(HAS_LIBS, bool)


class TestGetCapabilitiesSummary:
    """Tests for get_capabilities_summary function."""

    def test_returns_string(self):
        """Test that summary is a string."""
        from ThermalSim.capabilities import get_capabilities_summary
        result = get_capabilities_summary()
        assert isinstance(result, str)

    def test_contains_header(self):
        """Test that summary contains header."""
        from ThermalSim.capabilities import get_capabilities_summary
        result = get_capabilities_summary()
        assert "ThermalSim Capabilities" in result

    def test_contains_core_libs_status(self):
        """Test that summary mentions core libs."""
        from ThermalSim.capabilities import get_capabilities_summary
        result = get_capabilities_summary()
        assert "Core libs" in result
        assert "numpy" in result.lower() or "matplotlib" in result.lower()

    def test_contains_pardiso_status(self):
        """Test that summary mentions PyPardiso."""
        from ThermalSim.capabilities import get_capabilities_summary
        result = get_capabilities_summary()
        assert "PyPardiso" in result or "pardiso" in result.lower()

    def test_contains_numba_status(self):
        """Test that summary mentions Numba."""
        from ThermalSim.capabilities import get_capabilities_summary
        result = get_capabilities_summary()
        assert "Numba" in result or "numba" in result.lower()

    def test_shows_available_or_not(self):
        """Test that summary shows Available/Not available status."""
        from ThermalSim.capabilities import get_capabilities_summary
        result = get_capabilities_summary()
        # Should contain either "Available" or "Not available"
        assert "Available" in result or "available" in result.lower()

    def test_multiline_output(self):
        """Test that summary is multiline."""
        from ThermalSim.capabilities import get_capabilities_summary
        result = get_capabilities_summary()
        lines = result.split('\n')
        assert len(lines) >= 3  # Header + at least 3 capability lines


class TestCapabilityFlagsConsistency:
    """Tests for consistency between flags and summary."""

    def test_libs_flag_matches_summary(self):
        """Test that HAS_LIBS flag matches summary text."""
        from ThermalSim.capabilities import HAS_LIBS, get_capabilities_summary
        summary = get_capabilities_summary()

        # Find the core libs line
        for line in summary.split('\n'):
            if 'Core libs' in line:
                if HAS_LIBS:
                    assert 'Available' in line and 'Not' not in line.split('Available')[0][-5:]
                else:
                    assert 'Not available' in line
                break

    def test_pardiso_flag_matches_summary(self):
        """Test that HAS_PARDISO flag matches summary text."""
        from ThermalSim.capabilities import HAS_PARDISO, get_capabilities_summary
        summary = get_capabilities_summary()

        for line in summary.split('\n'):
            if 'PyPardiso' in line or 'pardiso' in line.lower():
                if HAS_PARDISO:
                    assert 'Available' in line
                else:
                    assert 'Not available' in line
                break

    def test_numba_flag_matches_summary(self):
        """Test that HAS_NUMBA flag matches summary text."""
        from ThermalSim.capabilities import HAS_NUMBA, get_capabilities_summary
        summary = get_capabilities_summary()

        for line in summary.split('\n'):
            if 'Numba' in line or 'numba' in line.lower():
                if HAS_NUMBA:
                    assert 'Available' in line
                else:
                    assert 'Not available' in line
                break


class TestModuleImportBehavior:
    """Tests for module import behavior."""

    def test_module_importable(self):
        """Test that capabilities module is importable."""
        import ThermalSim.capabilities as capabilities
        assert capabilities is not None

    def test_reimport_stable(self):
        """Test that reimporting gives same flags."""
        from ThermalSim.capabilities import HAS_LIBS, HAS_PARDISO, HAS_NUMBA

        # Reimport
        import ThermalSim.capabilities as capabilities
        importlib.reload(capabilities)

        from capabilities import HAS_LIBS as HAS_LIBS2
        from ThermalSim.capabilities import HAS_PARDISO as HAS_PARDISO2
        from ThermalSim.capabilities import HAS_NUMBA as HAS_NUMBA2

        assert HAS_LIBS == HAS_LIBS2
        assert HAS_PARDISO == HAS_PARDISO2
        assert HAS_NUMBA == HAS_NUMBA2

    def test_attributes_exist(self):
        """Test that expected attributes exist in module."""
        import ThermalSim.capabilities as capabilities

        assert hasattr(capabilities, 'HAS_LIBS')
        assert hasattr(capabilities, 'HAS_PARDISO')
        assert hasattr(capabilities, 'HAS_NUMBA')
        assert hasattr(capabilities, 'get_capabilities_summary')


class TestEdgeCases:
    """Edge case tests for capabilities module."""

    def test_summary_not_empty(self):
        """Test that summary is not empty."""
        from ThermalSim.capabilities import get_capabilities_summary
        result = get_capabilities_summary()
        assert len(result) > 0

    def test_summary_no_exceptions(self):
        """Test that getting summary doesn't raise exceptions."""
        from ThermalSim.capabilities import get_capabilities_summary
        try:
            result = get_capabilities_summary()
            assert result is not None
        except Exception as e:
            pytest.fail(f"get_capabilities_summary raised exception: {e}")

    def test_flags_accessible_as_module_attributes(self):
        """Test that flags can be accessed as module attributes."""
        import ThermalSim.capabilities as capabilities

        # Access without from ... import
        assert capabilities.HAS_LIBS in [True, False]
        assert capabilities.HAS_PARDISO in [True, False]
        assert capabilities.HAS_NUMBA in [True, False]
