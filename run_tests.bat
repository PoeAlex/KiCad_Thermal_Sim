@echo off
REM ThermalSim Test Runner
REM Runs the test suite with proper mock injection

echo Running ThermalSim unit tests...
echo.

python -c "import sys; sys.path.insert(0, 'tests'); from mocks.pcbnew_mock import install_mock; from mocks.wx_mock import install_wx_mock; install_mock(); install_wx_mock(); sys.path.insert(0, '.'); import pytest; sys.exit(pytest.main(['-v', '--tb=short', 'tests/'] + sys.argv[1:]))" %*

echo.
echo Done.
