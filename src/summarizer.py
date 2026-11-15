"""
Note Summarization & Revision Cards Generator.

Implements:
1. Graph-based Extractive TextRank Summarization (PageRank on Sentence Similarity Matrix)
2. Frequency & Position Salience Scoring
3. Key Concept & Definition Extractor
4. Exam Revision Flashcard Generator
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional
import numpy as np


@dataclass
class Flashcard:
    question: str
    answer: str
    concept: str


@dataclass
class NoteSummaryResult:
    executive_summary: str
    key_points: List[str]
    definitions: Dict[str, str]
    flashcards: List[Flashcard]
    compression_ratio: float


class NoteSummarizer:
    """
    Summarizes student notes and produces revision flashcards using extractive NLP techniques.
    """

    STOPWORDS: Set[str] = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
        "by", "can", "did", "do", "does", "each", "few", "for", "from", "further", "had", "has",
        "have", "having", "he", "her", "here", "hers", "him", "his", "how", "i", "if", "in", "into",
        "is", "it", "its", "me", "more", "most", "my", "no", "nor", "not", "of", "off", "on", "once",
        "only", "or", "other", "our", "out", "over", "own", "same", "she", "so", "some", "such",
        "than", "that", "the", "their", "theirs", "them", "then", "there", "these", "they", "this",
        "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what",
        "when", "where", "which", "while", "who", "whom", "why", "with", "you", "your"
    }

    DEFINITION_PATTERNS = [
        re.compile(r"^([\w\s\-]+?)\s+(?:is defined as|refers to|is a|is an|stands for|denotes)\s+(.+)$", re.IGNORECASE),
        re.compile(r"^([\w\s\-]+?)\s*:\s*(.+)$", re.IGNORECASE),
        re.compile(r"^(?:Formula|Equation)\s*:\s*(.+)$", re.IGNORECASE),
    ]

    def __init__(self, damping: float = 0.85, max_iter: int = 50):
        self.damping = damping
        self.max_iter = max_iter

    def split_into_sentences(self, text: str) -> List[str]:
        """Splits note text into coherent sentences or bullet items."""
        # Replace newlines with periods if not already punctuated
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        sentences: List[str] = []

        for line in lines:
            # Strip markdown prefixes
            clean_line = re.sub(r"^#{1,6}\s*", "", line)
            clean_line = re.sub(r"^[\*\-\•]\s*", "", clean_line)
            clean_line = re.sub(r"^\d+[\.\)]\s*", "", clean_line)

            # Split by punctuation
            raw_sents = re.split(r"(?<=[.!?])\s+", clean_line)
            for s in raw_sents:
                s_strip = s.strip()
                if len(s_strip.split()) >= 3:
                    sentences.append(s_strip)

        return sentences

    def _tokenize_sentence(self, sentence: str) -> List[str]:
        """Extracts lowercase alphabetic words excluding stopwords."""
        words = re.findall(r"\b[a-zA-Z]{2,}\b", sentence.lower())
        return [w for w in words if w not in self.STOPWORDS]

    def _build_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """
        Builds cosine-like overlap similarity matrix between sentences.
        Sim(S_i, S_j) = |tokens(S_i) ∩ tokens(S_j)| / (log(|tokens(S_i)|) + log(|tokens(S_j)|))
        """
        n = len(sentences)
        sim_mat = np.zeros((n, n), dtype=float)
        token_sets = [set(self._tokenize_sentence(s)) for s in sentences]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                len_i = len(token_sets[i])
                len_j = len(token_sets[j])
                if len_i == 0 or len_j == 0:
                    continue
                intersection = len(token_sets[i].intersection(token_sets[j]))
                denom = (np.log(len_i + 1.0) + np.log(len_j + 1.0))
                sim_mat[i][j] = intersection / max(1e-5, denom)

        return sim_mat

    def _pagerank(self, sim_mat: np.ndarray) -> np.ndarray:
        """Runs power iteration PageRank algorithm on sentence similarity matrix."""
        n = sim_mat.shape[0]
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([1.0])

        # Normalize rows to create transition probability matrix
        row_sums = sim_mat.sum(axis=1, keepdims=True)
        # Avoid division by zero for isolated nodes
        norm_mat = np.divide(sim_mat, row_sums, out=np.zeros_like(sim_mat), where=row_sums != 0)

        # PageRank vector
        v = np.ones(n, dtype=float) / n
        teleport = np.ones(n, dtype=float) / n

        for _ in range(self.max_iter):
            v_next = self.damping * norm_mat.T.dot(v) + (1.0 - self.damping) * teleport
            # Handle dangling nodes
            diff = np.linalg.norm(v_next - v, 1)
            v = v_next
            if diff < 1e-4:
                break

        return v

    def extract_definitions(self, text: str) -> Dict[str, str]:
        """Extracts definitions, formulas, and acronym expansions."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        definitions: Dict[str, str] = {}

        for line in lines:
            clean = re.sub(r"^[\*\-\•\d\.\)]\s*", "", line)
            for pat in self.DEFINITION_PATTERNS:
                match = pat.match(clean)
                if match:
                    term = match.group(1).strip()
                    meaning = match.group(2).strip()
                    if 2 <= len(term.split()) <= 5 and len(meaning.split()) >= 3:
                        definitions[term.title()] = meaning
                    break

        return definitions

    def generate_flashcards(self, definitions: Dict[str, str], top_sentences: List[str]) -> List[Flashcard]:
        """Transforms extracted key concepts and high-ranked sentences into revision cards."""
        cards: List[Flashcard] = []

        # 1. Cards from extracted definitions
        for term, meaning in definitions.items():
            cards.append(Flashcard(
                question=f"What is {term}?",
                answer=meaning,
                concept=term
            ))

        # 2. Cards from top high-yield sentences
        for sent in top_sentences[:3]:
            # Check if sentence has a colon or key concept
            if ":" in sent:
                parts = sent.split(":", 1)
                concept = parts[0].strip()
                ans = parts[1].strip()
                if len(concept.split()) <= 4 and len(ans.split()) >= 3:
                    cards.append(Flashcard(
                        question=f"Explain {concept}:",
                        answer=ans,
                        concept=concept
                    ))
            elif any(k in sent.lower() for k in ["algorithm", "formula", "method", "protocol", "model", "layer", "process", "system"]):
                cards.append(Flashcard(
                    question=f"Key Concept: {sent[:45]}...",
                    answer=sent,
                    concept="Core Concept"
                ))
            else:
                cards.append(Flashcard(
                    question=f"What is the significance of: {sent[:35]}...?",
                    answer=sent,
                    concept="Study Note"
                ))

        return cards

    def summarize(self, text: str, max_summary_sentences: int = 3) -> NoteSummaryResult:
        """
        Executes end-to-end extractive summarization:
        - TextRank PageRank scoring
        - Position and salience weighting
        - Definition and formula extraction
        - Interactive flashcard generation
        """
        sentences = self.split_into_sentences(text)
        if not sentences:
            return NoteSummaryResult(
                executive_summary="No sufficient text found to summarize.",
                key_points=[],
                definitions={},
                flashcards=[],
                compression_ratio=0.0
            )

        if len(sentences) <= max_summary_sentences:
            selected_sents = sentences
        else:
            sim_mat = self._build_similarity_matrix(sentences)
            scores = self._pagerank(sim_mat)

            # Boost first sentence (often topic header in notes)
            scores[0] *= 1.25

            ranked_indices = np.argsort(scores)[::-1][:max_summary_sentences]
            # Sort chronologically to preserve note flow
            sorted_indices = sorted(ranked_indices)
            selected_sents = [sentences[i] for i in sorted_indices]

        executive_summary = " ".join(selected_sents)
        definitions = self.extract_definitions(text)
        flashcards = self.generate_flashcards(definitions, selected_sents)

        orig_word_count = len(text.split())
        summary_word_count = len(executive_summary.split())
        compression = round((1.0 - (summary_word_count / max(1, orig_word_count))) * 100.0, 1)

        return NoteSummaryResult(
            executive_summary=executive_summary,
            key_points=selected_sents,
            definitions=definitions,
            flashcards=flashcards,
            compression_ratio=max(0.0, compression)
        )
