"""Tests for dubalot.translator."""

from unittest.mock import MagicMock, patch

import pytest

from dubalot.transcriber import Segment
from dubalot.translator import translate


class TestTranslate:
    @patch("dubalot.translator.GoogleTranslator")
    def test_translates_text(self, MockGoogleTranslator):
        mock_tr = MagicMock()
        mock_tr.translate.side_effect = lambda text: f"[translated] {text}"
        MockGoogleTranslator.return_value = mock_tr

        segments = [
            Segment(0.0, 1.0, "Hola", "es"),
            Segment(1.0, 2.0, "Mundo", "es"),
        ]
        result = translate(segments, target_language="en")

        assert len(result) == 2
        assert result[0].text == "[translated] Hola"
        assert result[1].text == "[translated] Mundo"

    @patch("dubalot.translator.GoogleTranslator")
    def test_preserves_timing(self, MockGoogleTranslator):
        mock_tr = MagicMock()
        mock_tr.translate.return_value = "Hello"
        MockGoogleTranslator.return_value = mock_tr

        segments = [Segment(1.5, 3.0, "Hola", "es")]
        result = translate(segments, target_language="en")

        assert result[0].start == 1.5
        assert result[0].end == 3.0

    @patch("dubalot.translator.GoogleTranslator")
    def test_updates_language(self, MockGoogleTranslator):
        mock_tr = MagicMock()
        mock_tr.translate.return_value = "Hello"
        MockGoogleTranslator.return_value = mock_tr

        segments = [Segment(0.0, 1.0, "Hola", "es")]
        result = translate(segments, target_language="en")

        assert result[0].language == "en"

    @patch("dubalot.translator.GoogleTranslator")
    def test_empty_segments(self, MockGoogleTranslator):
        MockGoogleTranslator.return_value = MagicMock()
        result = translate([], target_language="en")
        assert result == []

    @patch("dubalot.translator.GoogleTranslator")
    def test_skips_empty_text(self, MockGoogleTranslator):
        mock_tr = MagicMock()
        MockGoogleTranslator.return_value = mock_tr

        segments = [Segment(0.0, 0.5, "   ", "es")]
        result = translate(segments, target_language="en")

        mock_tr.translate.assert_not_called()
        assert result[0].text == "   "

    @patch("dubalot.translator.GoogleTranslator")
    def test_uses_auto_source(self, MockGoogleTranslator):
        mock_tr = MagicMock()
        mock_tr.translate.return_value = "Bonjour"
        MockGoogleTranslator.return_value = mock_tr

        translate([Segment(0.0, 1.0, "Hello", "en")], target_language="fr")
        MockGoogleTranslator.assert_called_once_with(source="auto", target="fr")
