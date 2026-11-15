"""
Unit tests for NoteSearchEngine.
"""

import pytest
from src.search_engine import NoteSearchEngine, NoteDocument, SearchResult


@pytest.fixture
def search_engine():
    engine = NoteSearchEngine()
    engine.add_document(NoteDocument(
        id="note_1",
        title="Binary Search Trees",
        content="A Binary Search Tree (BST) is a binary tree where left subtree has keys smaller than root.",
        subject="Data Structures",
        tags=["tree", "algorithms", "exam"]
    ))
    engine.add_document(NoteDocument(
        id="note_2",
        title="Dijkstra Shortest Path",
        content="Dijkstra algorithm calculates the shortest path from single source vertex in weighted graph.",
        subject="Algorithms",
        tags=["graphs", "greedy", "important"]
    ))
    engine.add_document(NoteDocument(
        id="note_3",
        title="Deadlock Handling in OS",
        content="Four conditions for deadlock: Mutual Exclusion, Hold and Wait, No Preemption, Circular Wait.",
        subject="Operating Systems",
        tags=["process", "deadlock", "exam"]
    ))
    return engine


def test_tokenize():
    tokens = NoteSearchEngine.tokenize("Dijkstra's Algorithm, O(V + E log V)!")
    assert "dijkstra" in tokens
    assert "algorithm" in tokens
    assert "s" in tokens or "log" in tokens


def test_exact_search(search_engine):
    results = search_engine.search("Dijkstra")
    assert len(results) > 0
    assert results[0].note.id == "note_2"
    assert "dijkstra" in results[0].matched_terms
    assert "<mark>" in results[0].snippet


def test_subject_filter(search_engine):
    results = search_engine.search("exam", subject_filter="Operating Systems")
    assert len(results) == 1
    assert results[0].note.id == "note_3"


def test_tag_filter(search_engine):
    results = search_engine.search("tree", tag_filter="exam")
    assert len(results) == 1
    assert results[0].note.id == "note_1"


def test_fuzzy_ocr_typo_search(search_engine):
    # Search for "dijkstr4" or "tre"
    results = search_engine.search("dijkstr", enable_fuzzy=True)
    assert len(results) > 0
    assert results[0].note.id == "note_2"


def test_empty_query_lists_all(search_engine):
    results = search_engine.search("")
    assert len(results) == 3


def test_json_serialization(search_engine):
    json_str = search_engine.to_json()
    assert "Binary Search Trees" in json_str

    new_engine = NoteSearchEngine()
    new_engine.load_json(json_str)
    assert len(new_engine.documents) == 3
    res = new_engine.search("Deadlock")
    assert len(res) == 1
    assert res[0].note.id == "note_3"
