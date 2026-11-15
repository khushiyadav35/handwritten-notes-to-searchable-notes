"""
Search & Retrieval Engine for Digital Notes.

Implements:
1. Inverted Index with Tokenization & Stopword Filtering
2. Okapi BM25 Ranking Algorithm (Term Frequency + IDF + Document Length Normalization)
3. Fuzzy Query Matching using Levenshtein Distance for OCR Typos
4. Subject and Tag Filtering
5. Contextual Snippet Generation with Search Highlights
"""

import math
import re
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Set, Optional, Tuple, Any


@dataclass
class NoteDocument:
    id: str
    title: str
    content: str
    subject: str = "General"
    tags: List[str] = field(default_factory=list)
    date_created: str = ""
    source_image: Optional[str] = None
    confidence: float = 90.0
    summary: str = ""


@dataclass
class SearchResult:
    note: NoteDocument
    score: float
    matched_terms: List[str]
    snippet: str


class NoteSearchEngine:
    """
    Search engine powered by an Inverted Index, BM25 ranking, and Fuzzy matching.
    """

    # Common English stopwords to ignore in pure keyword scoring
    STOPWORDS: Set[str] = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
        "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
        "but", "by", "can", "cannot", "could", "did", "do", "does", "doing", "down", "during", "each",
        "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers",
        "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it",
        "its", "itself", "let's", "me", "more", "most", "my", "myself", "no", "nor", "not", "of",
        "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over",
        "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
        "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
        "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while",
        "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
    }

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        k1: Controls term frequency saturation.
        b: Controls document length normalization penalization.
        """
        self.k1 = k1
        self.b = b
        self.documents: Dict[str, NoteDocument] = {}
        # Inverted index: word -> {doc_id: term_frequency}
        self.inverted_index: Dict[str, Dict[str, int]] = {}
        # Document lengths in words
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length: float = 0.0

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Splits text into lowercase alphanumeric tokens."""
        clean = re.sub(r"[^\w\s]", " ", text.lower())
        return [tok for tok in clean.split() if tok]

    def add_document(self, doc: NoteDocument):
        """Indexes a note document."""
        self.documents[doc.id] = doc
        combined_text = f"{doc.title} {doc.subject} {' '.join(doc.tags)} {doc.content}"
        tokens = self.tokenize(combined_text)

        self.doc_lengths[doc.id] = len(tokens)
        self.avg_doc_length = sum(self.doc_lengths.values()) / max(1, len(self.documents))

        # Count frequencies for this doc
        tf_map: Dict[str, int] = {}
        for token in tokens:
            tf_map[token] = tf_map.get(token, 0) + 1

        # Update inverted index
        for token, count in tf_map.items():
            if token not in self.inverted_index:
                self.inverted_index[token] = {}
            self.inverted_index[token][doc.id] = count

    def remove_document(self, doc_id: str):
        """Removes a document from the index."""
        if doc_id in self.documents:
            del self.documents[doc_id]
        if doc_id in self.doc_lengths:
            del self.doc_lengths[doc_id]

        for token in list(self.inverted_index.keys()):
            if doc_id in self.inverted_index[token]:
                del self.inverted_index[token][doc_id]
                if not self.inverted_index[token]:
                    del self.inverted_index[token]

        self.avg_doc_length = (
            sum(self.doc_lengths.values()) / max(1, len(self.documents))
            if self.documents else 0.0
        )

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Computes edit distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _expand_query_fuzzy(self, query_tokens: List[str], max_edit_distance: int = 1) -> Dict[str, float]:
        """
        Expands query terms to include similar words in vocabulary to handle OCR typos.
        Returns dict of {matched_vocab_term: similarity_weight}.
        """
        expanded: Dict[str, float] = {}
        vocab = list(self.inverted_index.keys())

        for q_tok in query_tokens:
            expanded[q_tok] = 1.0
            # If word is long enough, search for near-matches
            if len(q_tok) >= 4:
                for v_tok in vocab:
                    if v_tok == q_tok:
                        continue
                    if abs(len(v_tok) - len(q_tok)) <= max_edit_distance:
                        dist = self._levenshtein_distance(q_tok, v_tok)
                        if dist <= max_edit_distance:
                            # Discount fuzzy match slightly
                            expanded[v_tok] = 0.75

        return expanded

    def _generate_snippet(self, content: str, matched_terms: List[str], window: int = 160) -> str:
        """Finds the best context snippet containing matched keywords with <mark> tags."""
        lines = content.splitlines()
        best_line = ""
        best_match_count = 0

        # Find line with most term hits
        for line in lines:
            line_lower = line.lower()
            hits = sum(1 for term in matched_terms if term in line_lower)
            if hits > best_match_count:
                best_match_count = hits
                best_line = line.strip()

        if not best_line and lines:
            best_line = lines[0].strip()

        if len(best_line) > window:
            best_line = best_line[:window] + "..."

        # Highlight terms
        highlighted = best_line
        for term in matched_terms:
            pattern = re.compile(rf"(\b{re.escape(term)}\b)", re.IGNORECASE)
            highlighted = pattern.sub(r"<mark>\1</mark>", highlighted)

        return highlighted

    def search(
        self,
        query: str,
        subject_filter: Optional[str] = None,
        tag_filter: Optional[str] = None,
        enable_fuzzy: bool = True,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Executes BM25 search with optional fuzzy matching and metadata filters.
        """
        if not query or not query.strip():
            # Return all filtered notes sorted by id
            results = []
            for doc in self.documents.values():
                if subject_filter and subject_filter.lower() != "all" and doc.subject.lower() != subject_filter.lower():
                    continue
                if tag_filter and tag_filter.lower() not in [t.lower() for t in doc.tags]:
                    continue
                results.append(SearchResult(
                    note=doc,
                    score=1.0,
                    matched_terms=[],
                    snippet=doc.content[:160] + "..." if len(doc.content) > 160 else doc.content
                ))
            return results[:top_k]

        query_tokens = [t for t in self.tokenize(query) if t not in self.STOPWORDS]
        if not query_tokens:
            query_tokens = self.tokenize(query)

        term_weights = self._expand_query_fuzzy(query_tokens) if enable_fuzzy else {t: 1.0 for t in query_tokens}

        scores: Dict[str, float] = {}
        matched_terms_per_doc: Dict[str, List[str]] = {}

        N = len(self.documents)
        if N == 0:
            return []

        for term, weight in term_weights.items():
            if term not in self.inverted_index:
                continue

            posting = self.inverted_index[term]
            df = len(posting)
            # Standard BM25 IDF formulation
            idf = math.log(1.0 + (N - df + 0.5) / (df + 0.5))

            for doc_id, tf in posting.items():
                doc = self.documents[doc_id]

                # Apply metadata filters
                if subject_filter and subject_filter.lower() != "all" and doc.subject.lower() != subject_filter.lower():
                    continue
                if tag_filter and tag_filter.lower() not in [t.lower() for t in doc.tags]:
                    continue

                doc_len = self.doc_lengths.get(doc_id, 1)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1.0, self.avg_doc_length)))
                term_score = weight * idf * ((tf * (self.k1 + 1.0)) / max(1e-5, denom))

                scores[doc_id] = scores.get(doc_id, 0.0) + term_score
                if doc_id not in matched_terms_per_doc:
                    matched_terms_per_doc[doc_id] = []
                matched_terms_per_doc[doc_id].append(term)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        search_results: List[SearchResult] = []

        for doc_id, score in ranked:
            doc = self.documents[doc_id]
            matched = list(set(matched_terms_per_doc.get(doc_id, [])))
            snippet = self._generate_snippet(doc.content, matched)
            search_results.append(SearchResult(
                note=doc,
                score=round(score, 3),
                matched_terms=matched,
                snippet=snippet
            ))

        return search_results

    def to_json(self) -> str:
        """Serializes indexed documents to a JSON string."""
        docs_list = [asdict(doc) for doc in self.documents.values()]
        return json.dumps(docs_list, indent=2)

    def load_json(self, json_str: str):
        """Loads and indexes documents from a JSON string."""
        docs_list = json.loads(json_str)
        self.documents.clear()
        self.inverted_index.clear()
        self.doc_lengths.clear()
        for d in docs_list:
            doc = NoteDocument(**d)
            self.add_document(doc)
