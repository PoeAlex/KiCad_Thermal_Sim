"""
Unit tests for build_pcm_package script.

Tests ZIP structure, metadata.json content, and CLI argument handling.
"""

import pytest
import os
import sys
import json
import zipfile
import hashlib


# Add project root to path so we can import the build script
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBuildPcmPackage:
    """Tests for the PCM package build script."""

    def _run_build(self, tmp_path, extra_args=None):
        """Helper: run the build script and return the ZIP path."""
        import subprocess
        cmd = [sys.executable, os.path.join(_project_root, "build_pcm_package.py"),
               "--output-dir", str(tmp_path)]
        if extra_args:
            cmd.extend(extra_args)
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=_project_root)
        return result

    def test_build_creates_zip(self, tmp_path):
        """Test that the build script creates a ZIP file."""
        result = self._run_build(tmp_path)
        assert result.returncode == 0, f"Build failed: {result.stderr}"
        zips = list(tmp_path.glob("*.zip"))
        assert len(zips) == 1

    def test_zip_contains_metadata(self, tmp_path):
        """Test that ZIP contains metadata.json."""
        self._run_build(tmp_path)
        zips = list(tmp_path.glob("*.zip"))
        with zipfile.ZipFile(zips[0], 'r') as zf:
            assert "metadata.json" in zf.namelist()

    def test_zip_contains_plugin_files(self, tmp_path):
        """Test that ZIP contains all plugin source files."""
        self._run_build(tmp_path)
        zips = list(tmp_path.glob("*.zip"))
        with zipfile.ZipFile(zips[0], 'r') as zf:
            names = zf.namelist()
            assert "plugins/__init__.py" in names
            assert "plugins/thermal_plugin.py" in names
            assert "plugins/capabilities.py" in names
            assert "plugins/thermal_solver.py" in names
            assert "plugins/dependency_installer.py" in names

    def test_metadata_valid_json(self, tmp_path):
        """Test that metadata.json is valid JSON."""
        self._run_build(tmp_path)
        zips = list(tmp_path.glob("*.zip"))
        with zipfile.ZipFile(zips[0], 'r') as zf:
            meta = json.loads(zf.read("metadata.json"))
            assert isinstance(meta, dict)

    def test_metadata_has_required_fields(self, tmp_path):
        """Test that metadata.json has PCM-required fields."""
        self._run_build(tmp_path)
        zips = list(tmp_path.glob("*.zip"))
        with zipfile.ZipFile(zips[0], 'r') as zf:
            meta = json.loads(zf.read("metadata.json"))
            assert "$schema" in meta
            assert "identifier" in meta
            assert "name" in meta
            assert "versions" in meta
            assert "author" in meta
            assert "resources" in meta

    def test_metadata_identifier(self, tmp_path):
        """Test metadata identifier."""
        self._run_build(tmp_path)
        zips = list(tmp_path.glob("*.zip"))
        with zipfile.ZipFile(zips[0], 'r') as zf:
            meta = json.loads(zf.read("metadata.json"))
            assert meta["identifier"] == "com.github.poealex.kicad-thermal-sim"

    def test_metadata_author(self, tmp_path):
        """Test metadata author."""
        self._run_build(tmp_path)
        zips = list(tmp_path.glob("*.zip"))
        with zipfile.ZipFile(zips[0], 'r') as zf:
            meta = json.loads(zf.read("metadata.json"))
            assert meta["author"]["name"] == "Alex Poe"

    def test_version_override(self, tmp_path):
        """Test --version CLI override."""
        self._run_build(tmp_path, ["--version", "1.2.3"])
        zips = list(tmp_path.glob("*.zip"))
        assert "1.2.3" in zips[0].name
        with zipfile.ZipFile(zips[0], 'r') as zf:
            meta = json.loads(zf.read("metadata.json"))
            assert meta["versions"][0]["version"] == "1.2.3"

    def test_output_sha256(self, tmp_path):
        """Test that build output includes SHA-256 hash."""
        result = self._run_build(tmp_path)
        assert "SHA-256" in result.stdout or "sha256" in result.stdout.lower()

    def test_sha256_matches_file(self, tmp_path):
        """Test that printed SHA-256 matches the actual file."""
        result = self._run_build(tmp_path)
        zips = list(tmp_path.glob("*.zip"))
        actual_hash = hashlib.sha256(zips[0].read_bytes()).hexdigest()
        assert actual_hash in result.stdout

    def test_no_test_files_in_zip(self, tmp_path):
        """Test that test files are not included in the ZIP."""
        self._run_build(tmp_path)
        zips = list(tmp_path.glob("*.zip"))
        with zipfile.ZipFile(zips[0], 'r') as zf:
            names = zf.namelist()
            for name in names:
                assert "test_" not in name
                assert "conftest" not in name
                assert "mocks/" not in name

    def test_no_settings_json_in_zip(self, tmp_path):
        """Test that settings file is not included."""
        self._run_build(tmp_path)
        zips = list(tmp_path.glob("*.zip"))
        with zipfile.ZipFile(zips[0], 'r') as zf:
            names = zf.namelist()
            for name in names:
                assert "last_settings" not in name

    def test_metadata_schema_url(self, tmp_path):
        """Test metadata schema URL."""
        self._run_build(tmp_path)
        zips = list(tmp_path.glob("*.zip"))
        with zipfile.ZipFile(zips[0], 'r') as zf:
            meta = json.loads(zf.read("metadata.json"))
            assert "kicad.org/pcm/schemas/v1" in meta["$schema"]
