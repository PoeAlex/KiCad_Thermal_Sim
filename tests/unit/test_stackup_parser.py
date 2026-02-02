"""
Unit tests for stackup_parser module.

This module tests the S-expression parsing and stackup extraction
from .kicad_pcb files.
"""

import os
import pytest
import tempfile

from ThermalSim.stackup_parser import (
    _sexpr_extract_from_index,
    _sexpr_extract_block,
    _sexpr_find_all_blocks,
    parse_stackup_from_board_file,
    format_stackup_report_um,
)

from tests.fixtures.sample_boards import (
    SIMPLE_2_LAYER_STACKUP,
    SIMPLE_4_LAYER_STACKUP,
    SIMPLE_6_LAYER_STACKUP,
    STACKUP_WITH_SUBLAYERS,
    NO_STACKUP_BOARD,
    generate_kicad_pcb_content,
)


class TestSexprExtractFromIndex:
    """Tests for _sexpr_extract_from_index helper function."""

    def test_simple_block(self):
        """Test extraction of simple balanced block."""
        s = "(simple block)"
        block, next_idx = _sexpr_extract_from_index(s, 0)
        assert block == "(simple block)"
        assert next_idx == len(s)

    def test_nested_blocks(self):
        """Test extraction with nested parentheses."""
        s = "(outer (inner1) (inner2))"
        block, next_idx = _sexpr_extract_from_index(s, 0)
        assert block == "(outer (inner1) (inner2))"
        assert next_idx == len(s)

    def test_deeply_nested(self):
        """Test deeply nested blocks."""
        s = "(a (b (c (d))))"
        block, next_idx = _sexpr_extract_from_index(s, 0)
        assert block == "(a (b (c (d))))"

    def test_start_at_offset(self):
        """Test extraction starting at non-zero index."""
        s = "prefix (block) suffix"
        block, next_idx = _sexpr_extract_from_index(s, 7)
        assert block == "(block)"
        assert next_idx == 14

    def test_unbalanced_returns_none(self):
        """Test that unbalanced parentheses return None."""
        s = "(unbalanced"
        block, next_idx = _sexpr_extract_from_index(s, 0)
        assert block is None
        assert next_idx is None

    def test_empty_block(self):
        """Test extraction of empty block."""
        s = "()"
        block, next_idx = _sexpr_extract_from_index(s, 0)
        assert block == "()"
        assert next_idx == 2


class TestSexprExtractBlock:
    """Tests for _sexpr_extract_block helper function."""

    def test_find_stackup(self):
        """Test finding stackup block."""
        s = "(kicad_pcb (stackup (layer x)) (other y))"
        block = _sexpr_extract_block(s, "stackup")
        assert block == "(stackup (layer x))"

    def test_find_general(self):
        """Test finding general block."""
        s = "(kicad_pcb (general (thickness 1.6)))"
        block = _sexpr_extract_block(s, "general")
        assert block == "(general (thickness 1.6))"

    def test_not_found(self):
        """Test when token is not found."""
        s = "(kicad_pcb (layers))"
        block = _sexpr_extract_block(s, "stackup")
        assert block is None

    def test_with_start_offset(self):
        """Test searching with start offset."""
        s = "(first (target)) (second (target))"
        block = _sexpr_extract_block(s, "target", start=17)
        assert block == "(target)"


class TestSexprFindAllBlocks:
    """Tests for _sexpr_find_all_blocks helper function."""

    def test_find_multiple_layers(self):
        """Test finding multiple layer blocks."""
        s = '(stackup (layer "F.Cu") (layer "B.Cu"))'
        blocks = _sexpr_find_all_blocks(s, "layer")
        assert len(blocks) == 2
        assert '(layer "F.Cu")' in blocks
        assert '(layer "B.Cu")' in blocks

    def test_find_no_matches(self):
        """Test when no matches found."""
        s = "(stackup (copper) (dielectric))"
        blocks = _sexpr_find_all_blocks(s, "layer")
        assert blocks == []

    def test_nested_same_token(self):
        """Test with nested blocks of same token."""
        s = "(outer (layer (layer inner)))"
        blocks = _sexpr_find_all_blocks(s, "layer")
        # Should find the outer layer block first
        assert len(blocks) >= 1


class TestParseStackupFromBoardFile:
    """Tests for parse_stackup_from_board_file function."""

    @pytest.fixture
    def mock_board_with_file(self, temp_dir):
        """Create a mock board object with file."""
        from tests.mocks.pcbnew_mock import MockBoard

        def create_board(content, filename="test.kicad_pcb"):
            filepath = os.path.join(temp_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            return MockBoard(filename=filepath)

        return create_board

    def test_2_layer_stackup(self, mock_board_with_file):
        """Test parsing 2-layer stackup."""
        board = mock_board_with_file(SIMPLE_2_LAYER_STACKUP)
        result = parse_stackup_from_board_file(board)

        assert "error" not in result
        assert result["board_thickness_mm"] == 1.6
        assert len(result["copper"]) == 2
        assert result["copper"][0]["name"] == "F.Cu"
        assert result["copper"][1]["name"] == "B.Cu"
        assert len(result["dielectric_gaps_mm"]) == 1

    def test_4_layer_stackup(self, mock_board_with_file):
        """Test parsing 4-layer stackup."""
        board = mock_board_with_file(SIMPLE_4_LAYER_STACKUP)
        result = parse_stackup_from_board_file(board)

        assert "error" not in result
        assert len(result["copper"]) == 4
        assert result["copper"][0]["name"] == "F.Cu"
        assert result["copper"][1]["name"] == "In1.Cu"
        assert result["copper"][2]["name"] == "In2.Cu"
        assert result["copper"][3]["name"] == "B.Cu"
        assert len(result["dielectric_gaps_mm"]) == 3

    def test_6_layer_stackup(self, mock_board_with_file):
        """Test parsing 6-layer stackup."""
        board = mock_board_with_file(SIMPLE_6_LAYER_STACKUP)
        result = parse_stackup_from_board_file(board)

        assert "error" not in result
        assert len(result["copper"]) == 6
        assert len(result["dielectric_gaps_mm"]) == 5

    def test_sublayer_thickness_accumulation(self, mock_board_with_file):
        """Test that sublayer thicknesses are accumulated."""
        board = mock_board_with_file(STACKUP_WITH_SUBLAYERS)
        result = parse_stackup_from_board_file(board)

        assert "error" not in result
        # Dielectric should have accumulated thickness of 0.1 + 0.1 = 0.2
        dielectrics = result.get("dielectrics", [])
        assert len(dielectrics) >= 1
        assert dielectrics[0]["thickness_mm"] == 0.2

    def test_no_stackup_error(self, mock_board_with_file):
        """Test error when no stackup found."""
        board = mock_board_with_file(NO_STACKUP_BOARD)
        result = parse_stackup_from_board_file(board)

        assert "error" in result
        assert "stackup" in result["error"].lower()

    def test_no_filename_error(self):
        """Test error when board has no filename."""
        from tests.mocks.pcbnew_mock import MockBoard

        board = MockBoard(filename="")
        result = parse_stackup_from_board_file(board)

        assert "error" in result
        assert "filename" in result["error"].lower() or "save" in result["error"].lower()

    def test_file_not_found_error(self):
        """Test error when file doesn't exist."""
        from tests.mocks.pcbnew_mock import MockBoard

        board = MockBoard(filename="/nonexistent/path/board.kicad_pcb")
        result = parse_stackup_from_board_file(board)

        assert "error" in result

    def test_copper_thickness_parsed(self, mock_board_with_file):
        """Test that copper thickness is correctly parsed."""
        board = mock_board_with_file(SIMPLE_2_LAYER_STACKUP)
        result = parse_stackup_from_board_file(board)

        assert result["copper"][0]["thickness_mm"] == 0.035
        assert result["copper"][1]["thickness_mm"] == 0.035

    def test_dielectric_gap_calculation(self, mock_board_with_file):
        """Test dielectric gap calculation between copper layers."""
        board = mock_board_with_file(SIMPLE_4_LAYER_STACKUP)
        result = parse_stackup_from_board_file(board)

        gaps = result["dielectric_gaps_mm"]
        assert gaps[0] == 0.2  # F.Cu -> In1.Cu
        assert gaps[1] == 1.0  # In1.Cu -> In2.Cu (core)
        assert gaps[2] == 0.2  # In2.Cu -> B.Cu

    def test_copper_ids_extracted(self, mock_board_with_file):
        """Test that copper layer IDs are extracted."""
        board = mock_board_with_file(SIMPLE_2_LAYER_STACKUP)
        result = parse_stackup_from_board_file(board)

        assert len(result["copper_ids"]) == 2
        assert 0 in result["copper_ids"]  # F.Cu
        assert 31 in result["copper_ids"]  # B.Cu

    def test_copper_order_preserved(self, mock_board_with_file):
        """Test that copper layer order is preserved (top to bottom)."""
        content = generate_kicad_pcb_content(layer_count=4)
        board = mock_board_with_file(content)
        result = parse_stackup_from_board_file(board)

        copper_names = [c["name"] for c in result["copper"]]
        assert copper_names[0] == "F.Cu"  # Top
        assert copper_names[-1] == "B.Cu"  # Bottom

    def test_generated_2_layer(self, mock_board_with_file):
        """Test generated 2-layer board content."""
        content = generate_kicad_pcb_content(
            layer_count=2,
            board_thickness_mm=1.6,
            copper_thickness_mm=0.035
        )
        board = mock_board_with_file(content)
        result = parse_stackup_from_board_file(board)

        assert "error" not in result
        assert len(result["copper"]) == 2

    def test_generated_4_layer(self, mock_board_with_file):
        """Test generated 4-layer board content."""
        content = generate_kicad_pcb_content(
            layer_count=4,
            board_thickness_mm=1.6
        )
        board = mock_board_with_file(content)
        result = parse_stackup_from_board_file(board)

        assert "error" not in result
        assert len(result["copper"]) == 4

    def test_generated_6_layer(self, mock_board_with_file):
        """Test generated 6-layer board content."""
        content = generate_kicad_pcb_content(
            layer_count=6,
            board_thickness_mm=2.0
        )
        board = mock_board_with_file(content)
        result = parse_stackup_from_board_file(board)

        assert "error" not in result
        assert len(result["copper"]) == 6


class TestFormatStackupReportUm:
    """Tests for format_stackup_report_um function."""

    def test_valid_stackup_report(self):
        """Test formatting a valid stackup."""
        stack = {
            "board_thickness_mm": 1.6,
            "copper": [
                {"name": "F.Cu", "thickness_mm": 0.035, "layer_id": 0},
                {"name": "B.Cu", "thickness_mm": 0.035, "layer_id": 31},
            ],
            "dielectric_gaps_mm": [1.53],
        }

        report = format_stackup_report_um(stack)

        assert "1.600 mm" in report  # Board thickness
        assert "F.Cu" in report
        assert "B.Cu" in report
        assert "35.0 um" in report  # Copper thickness in um
        assert "1530.0 um" in report  # Dielectric gap in um

    def test_error_stackup(self):
        """Test formatting an error stackup."""
        stack = {"error": "No stackup found"}
        report = format_stackup_report_um(stack)

        assert "No stackup found" in report

    def test_none_stackup(self):
        """Test formatting None stackup raises error (actual behavior)."""
        # Note: The actual implementation doesn't handle None properly
        # This test documents the current behavior
        with pytest.raises(AttributeError):
            format_stackup_report_um(None)

    def test_missing_thickness(self):
        """Test formatting with missing thickness values."""
        stack = {
            "board_thickness_mm": None,
            "copper": [
                {"name": "F.Cu", "thickness_mm": None, "layer_id": 0},
            ],
            "dielectric_gaps_mm": [],
        }

        report = format_stackup_report_um(stack)
        assert "n/a" in report

    def test_4_layer_report(self):
        """Test formatting 4-layer stackup."""
        stack = {
            "board_thickness_mm": 1.6,
            "copper": [
                {"name": "F.Cu", "thickness_mm": 0.035, "layer_id": 0},
                {"name": "In1.Cu", "thickness_mm": 0.035, "layer_id": 1},
                {"name": "In2.Cu", "thickness_mm": 0.035, "layer_id": 2},
                {"name": "B.Cu", "thickness_mm": 0.035, "layer_id": 31},
            ],
            "dielectric_gaps_mm": [0.2, 1.0, 0.2],
        }

        report = format_stackup_report_um(stack)

        assert "F.Cu" in report
        assert "In1.Cu" in report
        assert "In2.Cu" in report
        assert "B.Cu" in report
        # Check gap formatting
        assert "F.Cu -> In1.Cu" in report
        assert "200.0 um" in report  # 0.2 mm gap
        assert "1000.0 um" in report  # 1.0 mm core

    def test_layer_id_formatting(self):
        """Test that layer IDs are formatted correctly."""
        stack = {
            "copper": [
                {"name": "F.Cu", "thickness_mm": 0.035, "layer_id": 0},
                {"name": "B.Cu", "thickness_mm": 0.035, "layer_id": 31},
            ],
            "dielectric_gaps_mm": [1.53],
        }

        report = format_stackup_report_um(stack)

        assert "id=0" in report
        assert "id=31" in report
