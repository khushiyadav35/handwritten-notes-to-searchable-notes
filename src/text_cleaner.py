"""
Text Cleaner & Post-Processing Pipeline for Handwritten Notes.

Transforms noisy raw OCR output into clean, structured, human-readable markdown.
Key capabilities:
1. Reassembling hyphenated words split across lines
2. Heuristic correction of common OCR character confusions (e.g. '0' vs 'O', '1' vs 'l')
3. Stripping stray margin noise, specks, and binding artifacts
4. Structural formatting: Bullet points, markdown headings, and math symbols
"""

import re
from typing import Dict, List, Tuple, Optional


class NoteTextCleaner:
    """
    Cleans, fixes typos, and normalizes formatting from raw OCR text.
    """

    # Common OCR character substitutions in English engineering notes
    COMMON_OCR_REPLACEMENTS = [
        # Numbers inside alphabetic words
        (r"\b([a-zA-Z]+)0([a-zA-Z]+)\b", r"\g<1>o\g<2>"),
        (r"\b([a-zA-Z]+)1([a-zA-Z]+)\b", r"\g<1>l\g<2>"),
        (r"\b([a-zA-Z]+)5([a-zA-Z]+)\b", r"\g<1>s\g<2>"),
        (r"\b([a-zA-Z]+)8([a-zA-Z]+)\b", r"\g<1>b\g<2>"),
        (r"\b([a-zA-Z]+)3([a-zA-Z]+)\b", r"\g<1>e\g<2>"),
        # Common handwriting ligatures and merges
        (r"\bclata\b", "data"),
        (r"\bleaming\b", "learning"),
        (r"\bnetvvork\b", "network"),
        (r"\bmemoly\b", "memory"),
        (r"\balg0rithm\b", "algorithm"),
        (r"\ba1gorithm\b", "algorithm"),
        (r"\bpr0cess\b", "process"),
        (r"\bsyst3m\b", "system"),
    ]

    # Math and logic arrows / relations
    MATH_SYMBOLS = [
        (r"\s*<==>\s*", " ⟺ "),
        (r"\s*<=>\s*", " ⟺ "),
        (r"\s*==>\s*", " ⟹ "),
        (r"\s*<->\s*", " ↔ "),
        (r"\s*->\s*", " → "),
        (r"\s*<-\s*", " ← "),
        (r"\s*<=\s*", " ≤ "),
        (r"\s*>=\s*", " ≥ "),
        (r"\s*!=\s*", " ≠ "),
    ]

    def __init__(self):
        pass

    def unbreak_hyphenated_words(self, text: str) -> str:
        """
        Reassembles words split across line breaks with a trailing hyphen.
        e.g., 'transfor-\\nmation' -> 'transformation'.
        """
        return re.sub(r"(\b[a-zA-Z]+)-\s*\n\s*([a-zA-Z]+\b)", r"\1\2", text)

    def remove_margin_artifacts(self, text: str) -> str:
        """
        Removes stray margin specks, solitary dots, stray pipe characters,
        or isolated punctuation lines produced by notebook binding or margins.
        """
        lines = text.splitlines()
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()
            # Skip isolated noise symbols
            if stripped in [".", ",", "'", '"', "|", "~", "`", "_", "-", "—", "/", "\\", "^"]:
                continue
            # Remove left margin bars like "| Text" or "1 | Text"
            stripped = re.sub(r"^[\|\:\;]\s*", "", stripped)
            # Remove repeated dashes or underscores (drawn divider lines)
            if re.match(r"^[-_=\*~]{3,}$", stripped):
                cleaned_lines.append("---")
                continue

            cleaned_lines.append(stripped)

        return "\n".join(cleaned_lines)

    def fix_character_confusions(self, text: str) -> str:
        """
        Corrects OCR misrecognitions where digits and letters get swapped.
        """
        cleaned = text
        for pattern, replacement in self.COMMON_OCR_REPLACEMENTS:
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        return cleaned

    def normalize_math_symbols(self, text: str) -> str:
        """Standardizes arrows and relation operators into clean Unicode symbols."""
        cleaned = text
        for pattern, replacement in self.MATH_SYMBOLS:
            cleaned = re.sub(pattern, replacement, cleaned)
        return cleaned

    def structure_headings_and_bullets(self, text: str) -> str:
        """
        Identifies note headings, sub-headings, and formats bullet lists cleanly.
        """
        lines = text.splitlines()
        formatted_lines = []

        for line in lines:
            line_str = line.strip()
            if not line_str:
                formatted_lines.append("")
                continue

            # Check if line looks like a main title or chapter header
            if re.match(r"^(UNIT|CHAPTER|LECTURE|MODULE|NOTE|TOPIC)\s*[-:]?\s*\d*", line_str, re.IGNORECASE):
                formatted_lines.append(f"## {line_str}")
                continue

            # Check if line is in ALL CAPS and relatively short (likely a section heading)
            if line_str.isupper() and len(line_str.split()) <= 6 and not line_str.startswith("#"):
                formatted_lines.append(f"### {line_str.title()}")
                continue

            # Standardize bullet markers (*, •, o, >, -)
            bullet_match = re.match(r"^[\*\•o\>\-]\s+(.*)$", line_str)
            if bullet_match:
                formatted_lines.append(f"- {bullet_match.group(1)}")
                continue

            # Standardize numbered lists (1., 1), (1))
            num_match = re.match(r"^(\(?\d+[\.\)])\s+(.*)$", line_str)
            if num_match:
                formatted_lines.append(f"1. {num_match.group(2)}")
                continue

            formatted_lines.append(line_str)

        return "\n".join(formatted_lines)

    def consolidate_whitespace(self, text: str) -> str:
        """Removes excessive blank lines and spaces while preserving paragraph breaks."""
        # Replace 3 or more newlines with double newline
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Replace multiple spaces with single space within lines
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def clean(self, raw_text: str) -> Dict[str, str]:
        """
        Full cleaning pipeline:
        1. Hyphen unbreaking
        2. Margin noise removal
        3. Character confusion fix
        4. Math symbols normalization
        5. Structural headings & bullet alignment
        6. Whitespace consolidation
        
        Returns dict with both 'clean_text' and 'markdown_text'.
        """
        if not raw_text or not raw_text.strip():
            return {"clean_text": "", "markdown_text": ""}

        step1 = self.unbreak_hyphenated_words(raw_text)
        step2 = self.remove_margin_artifacts(step1)
        step3 = self.fix_character_confusions(step2)
        step4 = self.normalize_math_symbols(step3)
        step5 = self.structure_headings_and_bullets(step4)
        final_text = self.consolidate_whitespace(step5)

        return {
            "clean_text": final_text,
            "markdown_text": final_text
        }
