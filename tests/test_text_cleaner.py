"""
Unit tests for NoteTextCleaner.
"""

import pytest
from src.text_cleaner import NoteTextCleaner


@pytest.fixture
def cleaner():
    return NoteTextCleaner()


def test_unbreak_hyphenated_words(cleaner):
    raw = "This is an imple-\nmentation of Dijkstra's algo-\nrithm."
    result = cleaner.unbreak_hyphenated_words(raw)
    assert "implementation" in result
    assert "algorithm" in result
    assert "imple-" not in result


def test_remove_margin_artifacts(cleaner):
    raw = "|\n.\n---\nSome actual lecture note content.\n~\n|"
    result = cleaner.remove_margin_artifacts(raw)
    lines = [l.strip() for l in result.splitlines() if l.strip()]
    assert "Some actual lecture note content." in lines
    assert "." not in lines
    assert "|" not in lines


def test_fix_character_confusions(cleaner):
    raw = "The c0mputer uses an a1gorithm to process clata and leaming models."
    result = cleaner.fix_character_confusions(raw)
    assert "computer" in result
    assert "algorithm" in result
    assert "data" in result
    assert "learning" in result


def test_normalize_math_symbols(cleaner):
    raw = "A -> B and X <=> Y where P <= Q and M != N"
    result = cleaner.normalize_math_symbols(raw)
    assert "→" in result
    assert "⟺" in result
    assert "≤" in result
    assert "≠" in result


def test_structure_headings_and_bullets(cleaner):
    raw = "UNIT 3: OPERATING SYSTEMS\n* Process scheduling\n• Memory management\n1. Paging\n2) Segmentation"
    result = cleaner.structure_headings_and_bullets(raw)
    assert "## UNIT 3: OPERATING SYSTEMS" in result
    assert "- Process scheduling" in result
    assert "- Memory management" in result
    assert "1. Paging" in result


def test_full_clean_pipeline(cleaner):
    raw = (
        "|\n"
        "TOPIC 2: GRAPH ALGORITHMS\n"
        "Breadth First Search (BFS) is a graph traver-\nsal a1gorithm.\n"
        "* Node A -> Node B\n"
        ".\n"
        "Time complexity: O(V + E)\n"
    )
    res = cleaner.clean(raw)
    clean_text = res["clean_text"]
    assert "### Topic 2: Graph Algorithms" in clean_text or "## TOPIC 2: GRAPH ALGORITHMS" in clean_text
    assert "traversal" in clean_text
    assert "algorithm" in clean_text
    assert "→" in clean_text
    assert "." not in clean_text.splitlines()
