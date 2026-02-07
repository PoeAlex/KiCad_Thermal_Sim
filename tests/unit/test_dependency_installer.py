"""
Unit tests for dependency_installer module.

Tests the DependencyInstallDialog and its pip-based installation logic.
"""

import pytest
import sys
import subprocess
from unittest.mock import patch, MagicMock


class TestDependencyInstallDialogCreation:
    """Tests for dialog creation and UI."""

    def test_dialog_creates_without_error(self):
        """Test that dialog can be instantiated."""
        from ThermalSim.dependency_installer import DependencyInstallDialog
        missing = [("numpy", "numpy"), ("scipy", "scipy")]
        dlg = DependencyInstallDialog(None, missing)
        assert dlg is not None

    def test_dialog_stores_missing_packages(self):
        """Test that dialog stores the missing packages list."""
        from ThermalSim.dependency_installer import DependencyInstallDialog
        missing = [("numpy", "numpy"), ("matplotlib", "matplotlib")]
        dlg = DependencyInstallDialog(None, missing)
        assert dlg._missing == missing

    def test_dialog_with_single_package(self):
        """Test dialog with only one missing package."""
        from ThermalSim.dependency_installer import DependencyInstallDialog
        missing = [("scipy", "scipy")]
        dlg = DependencyInstallDialog(None, missing)
        assert len(dlg._missing) == 1

    def test_dialog_with_empty_list(self):
        """Test dialog with no missing packages."""
        from ThermalSim.dependency_installer import DependencyInstallDialog
        dlg = DependencyInstallDialog(None, [])
        assert dlg._missing == []

    def test_dialog_not_installing_initially(self):
        """Test that dialog is not in installing state initially."""
        from ThermalSim.dependency_installer import DependencyInstallDialog
        dlg = DependencyInstallDialog(None, [("numpy", "numpy")])
        assert dlg._installing is False


class TestInstallLogic:
    """Tests for the pip install subprocess logic."""

    def test_on_install_sets_installing_flag(self):
        """Test that clicking install sets the installing flag."""
        from ThermalSim.dependency_installer import DependencyInstallDialog
        missing = [("numpy", "numpy")]
        dlg = DependencyInstallDialog(None, missing)

        with patch.object(dlg, '_run_pip'):
            import threading
            with patch.object(threading, 'Thread') as mock_thread:
                mock_thread_instance = MagicMock()
                mock_thread.return_value = mock_thread_instance
                dlg._on_install(None)
                assert dlg._installing is True

    def test_on_install_prevents_double_install(self):
        """Test that a second install click is ignored while installing."""
        from ThermalSim.dependency_installer import DependencyInstallDialog
        missing = [("numpy", "numpy")]
        dlg = DependencyInstallDialog(None, missing)
        dlg._installing = True

        # Should return early without starting a thread
        import threading
        with patch.object(threading, 'Thread') as mock_thread:
            dlg._on_install(None)
            mock_thread.assert_not_called()

    def test_run_pip_success(self):
        """Test successful pip run."""
        from ThermalSim.dependency_installer import DependencyInstallDialog
        dlg = DependencyInstallDialog(None, [("numpy", "numpy")])

        mock_process = MagicMock()
        mock_process.stdout = iter(["Collecting numpy\n", "Successfully installed numpy\n"])
        mock_process.wait.return_value = None
        mock_process.returncode = 0

        with patch('subprocess.Popen', return_value=mock_process):
            with patch.object(dlg, '_on_success') as mock_success:
                with patch('wx.CallAfter', side_effect=lambda f, *a, **kw: f(*a, **kw)):
                    dlg._run_pip([sys.executable, "-m", "pip", "install", "numpy"])
                    mock_success.assert_called_once()

    def test_run_pip_failure(self):
        """Test failed pip run."""
        from ThermalSim.dependency_installer import DependencyInstallDialog
        dlg = DependencyInstallDialog(None, [("numpy", "numpy")])

        mock_process = MagicMock()
        mock_process.stdout = iter(["ERROR: Could not find package\n"])
        mock_process.wait.return_value = None
        mock_process.returncode = 1

        with patch('subprocess.Popen', return_value=mock_process):
            with patch.object(dlg, '_on_failure') as mock_failure:
                with patch('wx.CallAfter', side_effect=lambda f, *a, **kw: f(*a, **kw)):
                    dlg._run_pip([sys.executable, "-m", "pip", "install", "numpy"])
                    mock_failure.assert_called_once_with(1)

    def test_run_pip_exception(self):
        """Test pip run raising an exception."""
        from ThermalSim.dependency_installer import DependencyInstallDialog
        dlg = DependencyInstallDialog(None, [("numpy", "numpy")])

        with patch('subprocess.Popen', side_effect=OSError("No pip")):
            with patch.object(dlg, '_on_failure') as mock_failure:
                with patch('wx.CallAfter', side_effect=lambda f, *a, **kw: f(*a, **kw)):
                    dlg._run_pip([sys.executable, "-m", "pip", "install", "numpy"])
                    mock_failure.assert_called_once()
                    args = mock_failure.call_args[0]
                    assert args[0] == -1
                    assert "No pip" in args[1]


class TestCallbackHandlers:
    """Tests for success/failure callback handlers."""

    def test_on_success_resets_installing_flag(self):
        """Test that success handler resets the installing flag."""
        from ThermalSim.dependency_installer import DependencyInstallDialog
        dlg = DependencyInstallDialog(None, [("numpy", "numpy")])
        dlg._installing = True
        dlg._on_success()
        assert dlg._installing is False

    def test_on_failure_resets_installing_flag(self):
        """Test that failure handler resets the installing flag."""
        from ThermalSim.dependency_installer import DependencyInstallDialog
        dlg = DependencyInstallDialog(None, [("numpy", "numpy")])
        dlg._installing = True
        dlg._on_failure(1)
        assert dlg._installing is False

    def test_on_failure_with_error_message(self):
        """Test failure handler with error message."""
        from ThermalSim.dependency_installer import DependencyInstallDialog
        dlg = DependencyInstallDialog(None, [("numpy", "numpy")])
        dlg._installing = True
        dlg._on_failure(-1, "Permission denied")
        assert dlg._installing is False

    def test_append_log(self):
        """Test that append_log adds text to the log control."""
        from ThermalSim.dependency_installer import DependencyInstallDialog
        dlg = DependencyInstallDialog(None, [("numpy", "numpy")])
        dlg._append_log("test line\n")
        # Verify log control has content (mock TextCtrl stores value)
        assert "test line" in dlg._log.GetValue()


class TestPipCommand:
    """Tests for pip command construction."""

    def test_install_uses_sys_executable(self):
        """Test that install uses the current Python interpreter."""
        from ThermalSim.dependency_installer import DependencyInstallDialog
        dlg = DependencyInstallDialog(None, [("numpy", "numpy")])

        captured_cmd = []

        def capture_thread(*args, **kwargs):
            # Extract the cmd argument from Thread(target=..., args=(cmd,))
            if 'args' in kwargs:
                captured_cmd.extend(kwargs['args'][0])
            elif len(args) > 0:
                pass
            mock_t = MagicMock()
            return mock_t

        import threading
        with patch.object(threading, 'Thread', side_effect=capture_thread):
            dlg._on_install(None)

        assert captured_cmd[0] == sys.executable
        assert captured_cmd[1:3] == ["-m", "pip"]

    def test_install_includes_all_packages(self):
        """Test that all missing packages are in the pip command."""
        from ThermalSim.dependency_installer import DependencyInstallDialog
        missing = [("numpy", "numpy"), ("scipy", "scipy"), ("matplotlib", "matplotlib")]
        dlg = DependencyInstallDialog(None, missing)

        captured_cmd = []

        def capture_thread(*args, **kwargs):
            if 'args' in kwargs:
                captured_cmd.extend(kwargs['args'][0])
            mock_t = MagicMock()
            return mock_t

        import threading
        with patch.object(threading, 'Thread', side_effect=capture_thread):
            dlg._on_install(None)

        assert "numpy" in captured_cmd
        assert "scipy" in captured_cmd
        assert "matplotlib" in captured_cmd
