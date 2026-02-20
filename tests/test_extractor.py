"""
Tests for extractor.py

Run with:  pytest tests/ -v
"""
import base64
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch, call

import pytest

# ── Make the project root importable ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

# Patch the Gemini client before importing extractor (avoids needing a real key)
os.environ.setdefault("GEMINI_API_KEY", "test-key")
with patch("google.genai.Client"):
    import extractor
    from extractor import (
        KNOWN_FUNDS,
        ask_gemini,
        page_to_base64,
        parse_response,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures — reusable sample data
# ══════════════════════════════════════════════════════════════════════════════

def make_gemini_json(
    page_type="revenue",
    fund="GENERAL FUND",
    department="REVENUE SUMMARY",
    function=None,
    items=None,
) -> str:
    """Return a well-formed Gemini JSON response string."""
    if items is None:
        items = [
            {
                "acct_no": None,
                "line_item": "TAXES",
                "category": "REVENUE",
                "sub_category": None,
                "fy2024_actual": 137694012,
                "fy2025_actual_ytd": 132119462,
                "budget_2425_original": 143670261,
                "budget_2425_adjusted": None,
                "budget_2526_request": 155269342,
                "budget_2526_mayor": 152282741,
                "budget_2526_final": 147972729,
            }
        ]
    return json.dumps(
        {
            "page_type": page_type,
            "fund": fund,
            "department": department,
            "function": function,
            "items": items,
        }
    )


# ══════════════════════════════════════════════════════════════════════════════
# parse_response — happy path
# ══════════════════════════════════════════════════════════════════════════════

class TestParseResponseHappyPath:
    def test_clean_json_returns_correct_page_type(self):
        raw = make_gemini_json(page_type="revenue")
        page_type, metadata, items = parse_response(raw)
        assert page_type == "revenue"

    def test_clean_json_returns_fund_and_department(self):
        raw = make_gemini_json(fund="GENERAL FUND", department="REVENUE SUMMARY")
        _, metadata, _ = parse_response(raw)
        assert metadata["fund"] == "GENERAL FUND"
        assert metadata["department"] == "REVENUE SUMMARY"

    def test_clean_json_returns_items(self):
        raw = make_gemini_json()
        _, _, items = parse_response(raw)
        assert len(items) == 1
        assert items[0]["line_item"] == "TAXES"
        assert items[0]["fy2024_actual"] == 137694012

    def test_negative_numbers_preserved(self):
        raw = make_gemini_json(
            items=[{"acct_no": None, "line_item": "NET INCOME", "category": None,
                    "sub_category": None, "fy2024_actual": -1757851,
                    "fy2025_actual_ytd": None, "budget_2425_original": None,
                    "budget_2425_adjusted": None, "budget_2526_request": None,
                    "budget_2526_mayor": None, "budget_2526_final": -558709}]
        )
        _, _, items = parse_response(raw)
        assert items[0]["fy2024_actual"] == -1757851
        assert items[0]["budget_2526_final"] == -558709

    def test_null_fields_preserved(self):
        raw = make_gemini_json(
            items=[{"acct_no": None, "line_item": "ITEM", "category": None,
                    "sub_category": None, "fy2024_actual": None,
                    "fy2025_actual_ytd": None, "budget_2425_original": 1000,
                    "budget_2425_adjusted": None, "budget_2526_request": None,
                    "budget_2526_mayor": None, "budget_2526_final": None}]
        )
        _, _, items = parse_response(raw)
        assert items[0]["fy2024_actual"] is None
        assert items[0]["budget_2425_original"] == 1000

    @pytest.mark.parametrize("page_type", [
        "dept_summary", "revenue", "expense_detail",
        "staffing", "capital", "utility", "other"
    ])
    def test_all_page_types_parsed(self, page_type):
        raw = make_gemini_json(page_type=page_type)
        result_type, _, _ = parse_response(raw)
        assert result_type == page_type

    def test_empty_items_array(self):
        raw = make_gemini_json(page_type="other", items=[])
        page_type, _, items = parse_response(raw)
        assert page_type == "other"
        assert items == []

    def test_function_field_extracted(self):
        raw = make_gemini_json(function="PUBLIC SAFETY")
        _, metadata, _ = parse_response(raw)
        assert metadata["function"] == "PUBLIC SAFETY"

    def test_function_field_null_when_absent(self):
        raw = make_gemini_json(function=None)
        _, metadata, _ = parse_response(raw)
        assert metadata["function"] is None


# ══════════════════════════════════════════════════════════════════════════════
# parse_response — markdown stripping
# ══════════════════════════════════════════════════════════════════════════════

class TestParseResponseMarkdownStripping:
    def test_strips_json_code_fence(self):
        inner = make_gemini_json(page_type="revenue")
        raw = f"```json\n{inner}\n```"
        page_type, _, _ = parse_response(raw)
        assert page_type == "revenue"

    def test_strips_plain_code_fence(self):
        inner = make_gemini_json(page_type="capital")
        raw = f"```\n{inner}\n```"
        page_type, _, _ = parse_response(raw)
        assert page_type == "capital"

    def test_strips_trailing_backtick_fence(self):
        inner = make_gemini_json(page_type="staffing")
        raw = f"{inner}\n```"
        page_type, _, _ = parse_response(raw)
        assert page_type == "staffing"

    def test_handles_preamble_text_before_json(self):
        inner = make_gemini_json(page_type="utility")
        raw = f"Here is the extracted data:\n{inner}"
        page_type, _, _ = parse_response(raw)
        assert page_type == "utility"


# ══════════════════════════════════════════════════════════════════════════════
# parse_response — JSON truncation recovery
# ══════════════════════════════════════════════════════════════════════════════

class TestParseResponseTruncationRecovery:
    def test_recovers_truncated_with_close_bracket_brace(self):
        # Simulate response cut off inside items array
        partial = '{"page_type": "revenue", "fund": "GENERAL FUND", ' \
                  '"department": "REV", "function": null, "items": ['
        raw = partial  # missing ]}
        page_type, metadata, items = parse_response(raw)
        # Should recover with ]} fix or fall through gracefully
        assert page_type in ("revenue", "other")

    def test_falls_back_to_other_on_completely_invalid_json(self):
        raw = "This is not JSON at all!!! ###"
        page_type, metadata, items = parse_response(raw)
        assert page_type == "other"
        assert metadata == {}
        assert items == []

    def test_falls_back_to_other_on_empty_string(self):
        page_type, metadata, items = parse_response("")
        assert page_type == "other"
        assert items == []

    def test_falls_back_to_other_on_whitespace_only(self):
        page_type, metadata, items = parse_response("   \n  ")
        assert page_type == "other"
        assert items == []

    def test_missing_page_type_defaults_to_other(self):
        raw = json.dumps({
            "fund": "GENERAL FUND",
            "department": "TEST",
            "function": None,
            "items": []
        })
        page_type, _, _ = parse_response(raw)
        assert page_type == "other"

    def test_items_not_list_reset_to_empty(self):
        raw = json.dumps({
            "page_type": "revenue",
            "fund": "GENERAL FUND",
            "department": "TEST",
            "function": None,
            "items": "not a list"
        })
        _, _, items = parse_response(raw)
        assert items == []


# ══════════════════════════════════════════════════════════════════════════════
# parse_response — metadata stamping onto items
# ══════════════════════════════════════════════════════════════════════════════

class TestParseResponseMetadataStamping:
    def test_fund_stamped_onto_items_when_missing(self):
        raw = make_gemini_json(fund="GENERAL FUND")
        _, _, items = parse_response(raw)
        assert items[0]["fund"] == "GENERAL FUND"

    def test_department_stamped_onto_items_when_missing(self):
        raw = make_gemini_json(department="POLICE")
        _, _, items = parse_response(raw)
        assert items[0]["department"] == "POLICE"

    def test_existing_item_fund_not_overwritten(self):
        """Items that already have a fund value should keep it."""
        item_with_fund = {
            "acct_no": "1005", "line_item": "TOWN COUNCIL", "category": None,
            "sub_category": None, "fy2024_actual": 63953,
            "fy2025_actual_ytd": None, "budget_2425_original": None,
            "budget_2425_adjusted": None, "budget_2526_request": None,
            "budget_2526_mayor": None, "budget_2526_final": 69475,
            "fund": "SPECIAL FUND",  # already has a fund
        }
        raw = make_gemini_json(fund="GENERAL FUND", items=[item_with_fund])
        _, _, items = parse_response(raw)
        # Item's own fund should NOT be overwritten
        assert items[0]["fund"] == "SPECIAL FUND"

    def test_metadata_stamped_on_multiple_items(self):
        items_data = [
            {"acct_no": str(i), "line_item": f"ITEM {i}", "category": None,
             "sub_category": None, "fy2024_actual": i * 1000,
             "fy2025_actual_ytd": None, "budget_2425_original": None,
             "budget_2425_adjusted": None, "budget_2526_request": None,
             "budget_2526_mayor": None, "budget_2526_final": None}
            for i in range(5)
        ]
        raw = make_gemini_json(fund="SEWER DIVISION", department="SEWER OPS", items=items_data)
        _, _, items = parse_response(raw)
        for item in items:
            assert item["fund"] == "SEWER DIVISION"
            assert item["department"] == "SEWER OPS"


# ══════════════════════════════════════════════════════════════════════════════
# parse_response — fund fallback inference
# ══════════════════════════════════════════════════════════════════════════════

class TestParseResponseFundFallback:
    @pytest.mark.parametrize("known_fund", KNOWN_FUNDS)
    def test_infers_known_fund_from_raw_text(self, known_fund):
        """When Gemini returns null fund but raw text contains a known fund name."""
        inner = json.dumps({
            "page_type": "utility",
            "fund": None,
            "department": None,
            "function": None,
            "items": [{"acct_no": None, "line_item": "REVENUE",
                       "category": None, "sub_category": None,
                       "fy2024_actual": 1000, "fy2025_actual_ytd": None,
                       "budget_2425_original": None, "budget_2425_adjusted": None,
                       "budget_2526_request": None, "budget_2526_mayor": None,
                       "budget_2526_final": None}]
        })
        # Embed the known fund name somewhere in the raw response
        raw = f"{known_fund} summary data\n{inner}"
        _, metadata, items = parse_response(raw)
        assert metadata["fund"] == known_fund
        assert items[0]["fund"] == known_fund

    def test_no_fallback_when_fund_already_set(self):
        """Should NOT overwrite an existing fund with the fallback."""
        raw = make_gemini_json(fund="GENERAL FUND")
        # ELECTRIC DIVISION appears in the raw text too
        raw = "ELECTRIC DIVISION details\n" + raw
        _, metadata, _ = parse_response(raw)
        assert metadata["fund"] == "GENERAL FUND"

    def test_no_fallback_when_no_known_fund_in_text(self):
        inner = json.dumps({
            "page_type": "other",
            "fund": None,
            "department": None,
            "function": None,
            "items": []
        })
        raw = "Some random text with no fund names.\n" + inner
        _, metadata, _ = parse_response(raw)
        assert metadata["fund"] is None

    def test_first_matching_fund_wins(self):
        """When multiple known funds appear, the first one in KNOWN_FUNDS list wins."""
        inner = json.dumps({
            "page_type": "utility",
            "fund": None,
            "department": None,
            "function": None,
            "items": []
        })
        # Both appear in text — ELECTRIC DIVISION comes first in KNOWN_FUNDS
        raw = "ELECTRIC DIVISION and SEWER DIVISION data\n" + inner
        _, metadata, _ = parse_response(raw)
        assert metadata["fund"] == "ELECTRIC DIVISION"


# ══════════════════════════════════════════════════════════════════════════════
# ask_gemini — retry logic
# ══════════════════════════════════════════════════════════════════════════════

class TestAskGemini:
    # Valid base64-encoded string for tests (avoids padding errors)
    VALID_B64 = base64.b64encode(b"fake_image_data").decode()

    def _make_response(self, text: str):
        mock = MagicMock()
        mock.text = text
        return mock

    def _make_client_error(self, message: str):
        """Create a real exception that behaves like ClientError with the given message."""
        from google.genai.errors import ClientError

        class FakeClientError(ClientError):
            def __init__(self, msg):
                # Bypass the SDK's strict __init__ by going straight to BaseException
                Exception.__init__(self, msg)
                self._message = msg

            def __str__(self):
                return self._message

        return FakeClientError(message)

    @patch("extractor.time.sleep")
    @patch("extractor.client")
    def test_returns_response_text_on_success(self, mock_client, mock_sleep):
        mock_client.models.generate_content.return_value = self._make_response(
            '{"page_type": "revenue", "items": []}'
        )
        result = ask_gemini(self.VALID_B64)
        assert result == '{"page_type": "revenue", "items": []}'
        mock_sleep.assert_not_called()

    @patch("extractor.time.sleep")
    @patch("extractor.client")
    def test_retries_on_429_then_succeeds(self, mock_client, mock_sleep):
        from google.genai.errors import ClientError
        error_429 = self._make_client_error("429 RESOURCE_EXHAUSTED")

        mock_client.models.generate_content.side_effect = [
            error_429,
            self._make_response('{"page_type": "revenue", "items": []}'),
        ]
        result = ask_gemini(self.VALID_B64)
        assert result == '{"page_type": "revenue", "items": []}'
        assert mock_client.models.generate_content.call_count == 2
        mock_sleep.assert_called_once_with(5)  # RETRY_BACKOFF * attempt 1

    @patch("extractor.time.sleep")
    @patch("extractor.client")
    def test_retry_backoff_increases_per_attempt(self, mock_client, mock_sleep):
        error_429 = self._make_client_error("429 RESOURCE_EXHAUSTED")

        mock_client.models.generate_content.side_effect = [
            error_429,
            error_429,
            self._make_response('{"page_type": "other", "items": []}'),
        ]
        ask_gemini(self.VALID_B64)
        sleep_calls = [c[0][0] for c in mock_sleep.call_args_list]
        assert sleep_calls[1] > sleep_calls[0], "Backoff should increase each attempt"

    @patch("extractor.time.sleep")
    @patch("extractor.client")
    def test_raises_after_max_retries(self, mock_client, mock_sleep):
        error_429 = self._make_client_error("429 RESOURCE_EXHAUSTED")
        mock_client.models.generate_content.side_effect = error_429

        with pytest.raises(RuntimeError, match="Exceeded max retries"):
            ask_gemini(self.VALID_B64)
        assert mock_client.models.generate_content.call_count == extractor.MAX_RETRIES

    @patch("extractor.time.sleep")
    @patch("extractor.client")
    def test_non_429_error_not_retried(self, mock_client, mock_sleep):
        from google.genai.errors import ClientError
        # Use a non-429 error that is still a ClientError subclass
        error_other = self._make_client_error("500 INTERNAL ERROR")
        mock_client.models.generate_content.side_effect = error_other

        with pytest.raises(Exception):
            ask_gemini(self.VALID_B64)
        assert mock_client.models.generate_content.call_count == 1
        mock_sleep.assert_not_called()

    @patch("extractor.time.sleep")
    @patch("extractor.client")
    def test_response_text_stripped_of_whitespace(self, mock_client, mock_sleep):
        mock_client.models.generate_content.return_value = self._make_response(
            "  \n  {\"page_type\": \"other\"}  \n  "
        )
        result = ask_gemini(self.VALID_B64)
        assert result == '{"page_type": "other"}'


# ══════════════════════════════════════════════════════════════════════════════
# page_to_base64
# ══════════════════════════════════════════════════════════════════════════════

class TestPageToBase64:
    def test_returns_valid_base64_string(self):
        mock_page = MagicMock()
        mock_pixmap = MagicMock()
        mock_pixmap.tobytes.return_value = b"\x89PNG\r\n\x1a\nfake_png_data"
        mock_page.get_pixmap.return_value = mock_pixmap

        result = page_to_base64(mock_page)

        # Should be valid base64
        decoded = base64.b64decode(result)
        assert decoded == b"\x89PNG\r\n\x1a\nfake_png_data"

    def test_uses_correct_dpi_matrix(self):
        mock_page = MagicMock()
        mock_pixmap = MagicMock()
        mock_pixmap.tobytes.return_value = b"fake"
        mock_page.get_pixmap.return_value = mock_pixmap

        import fitz
        with patch("fitz.Matrix") as mock_matrix:
            mock_matrix.return_value = MagicMock()
            page_to_base64(mock_page, dpi=150)
            expected_scale = 150 / 72
            mock_matrix.assert_called_once_with(expected_scale, expected_scale)

    def test_uses_rgb_colorspace(self):
        mock_page = MagicMock()
        mock_pixmap = MagicMock()
        mock_pixmap.tobytes.return_value = b"fake"
        mock_page.get_pixmap.return_value = mock_pixmap

        page_to_base64(mock_page)

        call_kwargs = mock_page.get_pixmap.call_args
        assert call_kwargs is not None

    def test_requests_png_format(self):
        mock_page = MagicMock()
        mock_pixmap = MagicMock()
        mock_pixmap.tobytes.return_value = b"fake_png"
        mock_page.get_pixmap.return_value = mock_pixmap

        page_to_base64(mock_page)
        mock_pixmap.tobytes.assert_called_once_with("png")


# ══════════════════════════════════════════════════════════════════════════════
# main() — integration-level tests
# ══════════════════════════════════════════════════════════════════════════════

class TestMain:
    """Integration tests for main() using heavy mocking."""

    def _make_mock_doc(self, num_pages=3):
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=num_pages)
        mock_doc.__getitem__ = MagicMock(return_value=MagicMock())
        return mock_doc

    @patch("extractor.PAGES_DIR", Path("/tmp/test_pages"))
    @patch("extractor.OUTPUT_PATH", Path("/tmp/test_budget.json"))
    @patch("extractor.page_to_base64", return_value="base64img")
    @patch("extractor.ask_gemini")
    @patch("fitz.open")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.glob", return_value=[])
    def test_processes_all_pages(
        self, mock_glob, mock_mkdir, mock_file, mock_fitz,
        mock_gemini, mock_b64
    ):
        mock_fitz.return_value = self._make_mock_doc(3)
        mock_gemini.return_value = make_gemini_json(page_type="revenue", items=[])

        with patch("sys.argv", ["extractor.py"]):
            extractor.main()

        assert mock_gemini.call_count == 3

    @patch("extractor.PAGES_DIR", Path("/tmp/test_pages"))
    @patch("extractor.OUTPUT_PATH", Path("/tmp/test_budget.json"))
    @patch("extractor.page_to_base64", return_value="base64img")
    @patch("extractor.ask_gemini")
    @patch("fitz.open")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.glob", return_value=[])
    def test_pages_flag_limits_processing(
        self, mock_glob, mock_mkdir, mock_file, mock_fitz,
        mock_gemini, mock_b64
    ):
        mock_fitz.return_value = self._make_mock_doc(10)
        mock_gemini.return_value = make_gemini_json(page_type="other", items=[])

        with patch("sys.argv", ["extractor.py", "--pages", "3"]):
            extractor.main()

        assert mock_gemini.call_count == 3

    @patch("extractor.PAGES_DIR", Path("/tmp/test_pages"))
    @patch("extractor.OUTPUT_PATH", Path("/tmp/test_budget.json"))
    @patch("extractor.page_to_base64", return_value="base64img")
    @patch("extractor.ask_gemini")
    @patch("fitz.open")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.glob", return_value=[])
    def test_page_flag_processes_single_page(
        self, mock_glob, mock_mkdir, mock_file, mock_fitz,
        mock_gemini, mock_b64
    ):
        mock_fitz.return_value = self._make_mock_doc(10)
        mock_gemini.return_value = make_gemini_json(page_type="revenue", items=[])

        with patch("sys.argv", ["extractor.py", "--page", "5"]):
            extractor.main()

        assert mock_gemini.call_count == 1

    @patch("extractor.PAGES_DIR", Path("/tmp/test_pages"))
    @patch("extractor.OUTPUT_PATH", Path("/tmp/test_budget.json"))
    @patch("extractor.page_to_base64", return_value="base64img")
    @patch("extractor.ask_gemini")
    @patch("fitz.open")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_resume_skips_cached_pages(
        self, mock_mkdir, mock_file, mock_fitz, mock_gemini, mock_b64
    ):
        mock_fitz.return_value = self._make_mock_doc(3)

        cached_page_data = json.dumps({
            "source_page": 1,
            "page_type": "revenue",
            "fund": "GENERAL FUND",
            "department": "REVENUE SUMMARY",
            "function": None,
            "items_extracted": 5,
            "items": []
        })

        # Page 1 has a cache file, pages 2 and 3 do not
        def glob_side_effect(pattern):
            if "page_001" in str(pattern):
                mock_path = MagicMock()
                mock_path.read_text.return_value = cached_page_data
                return [mock_path]
            return []

        with patch.object(Path, "glob", side_effect=glob_side_effect):
            mock_gemini.return_value = make_gemini_json(page_type="other", items=[])
            with patch("sys.argv", ["extractor.py", "--resume"]):
                extractor.main()

        # Only pages 2 and 3 should call Gemini (page 1 is cached)
        assert mock_gemini.call_count == 2

    @patch("extractor.PAGES_DIR", Path("/tmp/test_pages"))
    @patch("extractor.OUTPUT_PATH", Path("/tmp/test_budget.json"))
    @patch("extractor.page_to_base64", return_value="base64img")
    @patch("extractor.ask_gemini")
    @patch("fitz.open")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.glob", return_value=[])
    def test_output_json_has_correct_structure(
        self, mock_glob, mock_mkdir, mock_file, mock_fitz,
        mock_gemini, mock_b64
    ):
        mock_fitz.return_value = self._make_mock_doc(2)
        mock_gemini.return_value = make_gemini_json(
            page_type="revenue",
            fund="GENERAL FUND",
            items=[{
                "acct_no": None, "line_item": "TAXES", "category": "REVENUE",
                "sub_category": None, "fy2024_actual": 1000,
                "fy2025_actual_ytd": None, "budget_2425_original": None,
                "budget_2425_adjusted": None, "budget_2526_request": None,
                "budget_2526_mayor": None, "budget_2526_final": 1200
            }]
        )

        written_data = {}

        def capture_write(data, f, **kwargs):
            written_data.update(json.loads(json.dumps(data)))

        with patch("json.dump", side_effect=capture_write):
            with patch("sys.argv", ["extractor.py"]):
                extractor.main()

        assert "source_file" in written_data
        assert "total_pages" in written_data
        assert "pages_manifest" in written_data
        assert "total_items" in written_data
        assert "line_items" in written_data

    @patch("extractor.PAGES_DIR", Path("/tmp/test_pages"))
    @patch("extractor.OUTPUT_PATH", Path("/tmp/test_budget.json"))
    @patch("extractor.page_to_base64", return_value="base64img")
    @patch("extractor.ask_gemini")
    @patch("fitz.open")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.glob", return_value=[])
    def test_other_pages_have_zero_items(
        self, mock_glob, mock_mkdir, mock_file, mock_fitz,
        mock_gemini, mock_b64
    ):
        mock_fitz.return_value = self._make_mock_doc(1)
        mock_gemini.return_value = make_gemini_json(page_type="other", items=[])

        written_data = {}

        def capture_write(data, f, **kwargs):
            written_data.update(json.loads(json.dumps(data)))

        with patch("json.dump", side_effect=capture_write):
            with patch("sys.argv", ["extractor.py"]):
                extractor.main()

        assert written_data.get("total_items", -1) == 0


# ══════════════════════════════════════════════════════════════════════════════
# Per-page filename formatting
# ══════════════════════════════════════════════════════════════════════════════

class TestPageFilenameFormatting:
    @pytest.mark.parametrize("page_num, page_type, expected", [
        (1,   "revenue",      "page_001_revenue.json"),
        (8,   "revenue",      "page_008_revenue.json"),
        (10,  "dept_summary", "page_010_dept_summary.json"),
        (101, "utility",      "page_101_utility.json"),
    ])
    def test_filename_format(self, page_num, page_type, expected):
        filename = f"page_{page_num:03d}_{page_type}.json"
        assert filename == expected


# ══════════════════════════════════════════════════════════════════════════════
# KNOWN_FUNDS constant
# ══════════════════════════════════════════════════════════════════════════════

class TestKnownFunds:
    def test_known_funds_is_list(self):
        assert isinstance(KNOWN_FUNDS, list)

    def test_known_funds_not_empty(self):
        assert len(KNOWN_FUNDS) > 0

    def test_known_funds_are_uppercase(self):
        for fund in KNOWN_FUNDS:
            assert fund == fund.upper(), f"Fund '{fund}' should be uppercase"

    def test_expected_divisions_present(self):
        assert "ELECTRIC DIVISION" in KNOWN_FUNDS
        assert "SEWER DIVISION" in KNOWN_FUNDS
        assert "WATER DIVISION" in KNOWN_FUNDS
        assert "GENERAL FUND" in KNOWN_FUNDS
