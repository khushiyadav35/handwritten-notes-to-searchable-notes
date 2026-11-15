"""
Unit tests for NoteSummarizer.
"""

import pytest
from src.summarizer import NoteSummarizer, NoteSummaryResult, Flashcard


@pytest.fixture
def summarizer():
    return NoteSummarizer()


@pytest.fixture
def sample_lecture_text():
    return (
        "# Operating Systems: Process Scheduling\n\n"
        "A Process is defined as a program in execution containing code and execution context.\n"
        "The CPU scheduler selects among the processes in memory that are ready to execute.\n"
        "First-Come First-Served is an algorithm that allocates CPU to the process requesting it first.\n"
        "Shortest Job First is an optimal scheduling algorithm that minimizes average waiting time.\n"
        "Round Robin scheduling is designed especially for time-sharing systems using a time quantum.\n"
        "Formula: Turnaround Time = Completion Time - Arrival Time.\n"
        "Formula: Waiting Time = Turnaround Time - Burst Time.\n"
    )


def test_sentence_splitting(summarizer, sample_lecture_text):
    sents = summarizer.split_into_sentences(sample_lecture_text)
    assert len(sents) >= 5
    assert any("CPU scheduler" in s for s in sents)


def test_definitions_extraction(summarizer, sample_lecture_text):
    defs = summarizer.extract_definitions(sample_lecture_text)
    assert len(defs) > 0
    assert any("Process" in k for k in defs.keys())


def test_summarization_output(summarizer, sample_lecture_text):
    res = summarizer.summarize(sample_lecture_text, max_summary_sentences=3)
    assert isinstance(res, NoteSummaryResult)
    assert len(res.key_points) <= 3
    assert len(res.executive_summary) > 20
    assert res.compression_ratio > 0.0
    assert len(res.flashcards) > 0


def test_flashcard_generation(summarizer, sample_lecture_text):
    res = summarizer.summarize(sample_lecture_text)
    assert len(res.flashcards) > 0
    first_card = res.flashcards[0]
    assert isinstance(first_card, Flashcard)
    assert len(first_card.question) > 0
    assert len(first_card.answer) > 0
