# NoteScribe: Handwritten Notes to Searchable Digital Notes

> An end-to-end Computer Science B.Tech capstone project designed to transform unorganized smartphone photos of handwritten notebooks and whiteboards into clean, fully searchable, indexed digital notes with automated summarization and exam revision flashcards.

---

## 🎯 Problem Statement
Students routinely photograph notebook pages, lecture chalkboards, and peer notes during class. However, these photos accumulate into hundreds of unorganized gallery images with uneven lighting, crumpled paper, skewed angles, and ruled notebook lines. Finding a specific concept before an exam is frustrating and time-consuming.

**NoteScribe** provides an automated computer vision and NLP pipeline:
1. **Adaptive Image Preprocessing**: Deskewing, shadow removal, adaptive binarization, and horizontal rule line suppression.
2. **Handwriting OCR Engine**: Accurate text extraction with word-level confidence evaluation.
3. **Contextual Text Cleaner**: Repairing broken hyphenated words, correcting OCR confusion artifacts, and structuring into bullet points and markdown headers.
4. **Search & Indexing Engine**: Inverted index with BM25 ranking, tag filtering, and fuzzy search tolerant to handwriting recognition errors.
5. **Key Takeaways & Summarization**: Extractive graph-based TextRank summarization, definition extraction, and interactive flashcards for quick revision.

---

## 🏗️ Project Architecture
```text
Raw Notebook Image (Phone / Scanner)
        │
        ▼
[Pre-Processing Pipeline]
  ├── Shadow Removal & Grayscale
  ├── Horizontal Rule Line Suppression (Morphological Kernel)
  ├── Adaptive Binarization (Otsu & Sauvola)
  └── Deskew & Perspective Normalization
        │
        ▼
[OCR Extraction Engine]
  ├── Tesseract LSTM Engine / Feature Extraction
  └── Confidence & Word-Level Scoring
        │
        ▼
[Text Cleaning & Normalization]
  ├── Cross-line Hyphen Reassembly
  ├── Heuristic Character Confusion Repair
  └── Structural Formatting (Headers, Bullets, Math Symbols)
        │
        ▼
[Storage & Indexing] ───► [BM25 & Fuzzy Search Engine]
        │
        ▼
[Extractive Summarizer] ──► [Key Definitions & Revision Cards]
```

---

## 🚀 Quick Setup & Installation

### Prerequisites
- Python 3.9+ installed
- Tesseract OCR engine (optional, fallback engine included for offline evaluation)

### Installation
```bash
# Clone the repository
git clone https://github.com/khushiyadav35/handwritten-notes-to-searchable-notes.git
cd handwritten-notes-to-searchable-notes

# Install dependencies
pip install -r requirements.txt

# Launch interactive Streamlit dashboard
streamlit run app/main.py
```

---

## 👩‍💻 Author
**Khushi Yadav**  
B.Tech Computer Science & Engineering  
GitHub: [@khushiyadav35](https://github.com/khushiyadav35)
