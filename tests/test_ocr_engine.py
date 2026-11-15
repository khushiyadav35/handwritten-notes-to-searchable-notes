"""
Unit tests for NoteOCREngine and OCRResult.
"""

import pytest
import numpy as np
from PIL import Image
from src.ocr_engine import NoteOCREngine, OCRResult, WordBox


def test_ocr_engine_initialization():
    engine = NoteOCREngine(psm=6)
    assert engine.psm == 6
    assert engine.oem == 3
    assert isinstance(engine.is_tesseract_installed(), bool)


def test_ocr_fallback_extraction():
    engine = NoteOCREngine()
    # Force fallback
    img = Image.new("RGB", (200, 100), color=(255, 255, 255))
    res = engine._run_fallback(img)

    assert isinstance(res, OCRResult)
    assert len(res.text) > 0
    assert res.line_count > 0
    assert res.avg_confidence >= 80.0
    assert len(res.words) > 0
    assert res.engine_used == "builtin-vision-fallback"


def test_ocr_custom_fallback_text():
    engine = NoteOCREngine()
    img = Image.new("RGB", (100, 50), color=(255, 255, 255))
    custom_text = "Binary Search Tree\nLeft child is smaller than root."
    res = engine.extract_text(img, fallback_text=custom_text)

    assert isinstance(res, OCRResult)
    assert "Binary Search Tree" in res.text
    assert res.line_count == 2
    assert len(res.words) == 9


def test_word_box_structure():
    wb = WordBox(text="Algorithm", confidence=94.5, x=10, y=20, width=50, height=15)
    assert wb.text == "Algorithm"
    assert wb.confidence == 94.5
    assert wb.width == 50
