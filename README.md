# NoteScribe: Handwritten Notes to Searchable Digital Notes

[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-4F46E5?style=for-the-badge&logo=github)](https://khushiyadav35.github.io/handwritten-notes-to-searchable-notes/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![Tesseract OCR](https://img.shields.io/badge/OCR-Tesseract%205-008080?style=for-the-badge)](https://github.com/tesseract-ocr/tesseract)
[![Tests Passing](https://img.shields.io/badge/Pytest-29%20Passed-success?style=for-the-badge&logo=pytest)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE.md)

> **B.Tech Computer Science & Engineering Final Year Capstone Project**  
> Developed by **Khushi Yadav** ([@khushiyadav35](https://github.com/khushiyadav35))  
> **Live Web Application:** [https://khushiyadav35.github.io/handwritten-notes-to-searchable-notes/](https://khushiyadav35.github.io/handwritten-notes-to-searchable-notes/)

---

## 📌 Problem Statement
Students photograph hundreds of notebook pages, peer notes, and lecture chalkboards during a semester. However:
1. **Unorganized & Lost:** Photos accumulate into phone galleries where finding a single concept before exams is nearly impossible.
2. **Poor Visual Quality:** Camera shadows, notebook ruled lines, skewed camera angles, and faint ink cause standard OCR engines to fail.
3. **No Semantic Search:** Image filenames like `IMG_20261012_104523.jpg` lack searchable metadata or topic tags.
4. **Exam Time Crunch:** Students spend hours rereading full pages instead of quickly revising core formulas, definitions, and flashcards.

**NoteScribe** solves this through an end-to-end Computer Vision and Natural Language Processing pipeline that digitizes, cleans, indexes, and summarizes handwritten study notes.

---

## 🌟 Key Features

| Module | Technical Implementation | Practical Student Benefit |
| :--- | :--- | :--- |
| **Adaptive Preprocessing** | Gaussian shadow division, Sauvola local binarization ($k=0.2$), horizontal 1D opening, Hough projection deskew | Converts uneven, shadowed camera photos into crisp, high-contrast black-and-white text |
| **Ruled Line Suppression** | Horizontal morphological opening ($1 \times 31$ kernel) | Strips notebook ruling so lines are never mistaken for dashes, hyphens, or minus signs |
| **OCR Extraction** | Tesseract 5 LSTM engine with `--psm 6` and per-word confidence scoring | Accurate character recognition with graceful fallback for offline environments |
| **Text Post-Processing** | Hyphen unbreaking regex, character confusion matrix (`pr0cess` $\to$ `process`), Markdown headers | Automatically structures messy OCR text into clean Markdown notes with bullet points |
| **Smart Search** | Okapi BM25 ranking ($k_1=1.5, b=0.75$) + Levenshtein distance $\le 1$ | Instant search across all notes; tolerant to handwriting OCR spelling errors |
| **AI Summarization** | Graph-based TextRank PageRank with sentence similarity matrix | Generates executive TL;DR and high-yield study bullets in milliseconds |
| **Exam Flashcards** | Deterministic definition pattern extractor | Automatically turns concepts and formulas into interactive Q&A flashcards |
| **Export & Persistence** | LocalStorage + Markdown / JSON / PDF download | Students retain full ownership of their notes without external cloud lock-in |

---

## 🏗️ System Architecture

```text
               [Smartphone / Camera Note Capture]
                               │
                               ▼
    ┌──────────────────────────────────────────────────────┐
    │          STAGE 1: IMAGE PREPROCESSING                │
    │  • Gaussian Filter Background Division (Shadow Free) │
    │  • Horizontal Morphological Opening (Line Removal)   │
    │  • Projection Profile Variance (Deskewing)           │
    │  • Local Adaptive Sauvola Binarization               │
    └──────────────────────────┬───────────────────────────┘
                               │
                               ▼
    ┌──────────────────────────────────────────────────────┐
    │          STAGE 2: OCR EXTRACTION                     │
    │  • Tesseract LSTM Engine (PSM 6: Uniform Text Block) │
    │  • Word-Level Confidence Metric Calculation          │
    └──────────────────────────┬───────────────────────────┘
                               │
                               ▼
    ┌──────────────────────────────────────────────────────┐
    │          STAGE 3: TEXT CLEANER & FORMATTER           │
    │  • Cross-line Hyphen Reassembly                      │
    │  • OCR Confusion Repair (c0mputer -> computer)       │
    │  • Structural Heading (#) & Bullet Standardization   │
    └──────────────────────────┬───────────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
┌───────────────────────────────┐ ┌───────────────────────────────┐
│  STAGE 4: SEARCH & RETRIEVAL  │ │  STAGE 5: SUMMARIZER & CARDS  │
│ • Inverted Word Index         │ │ • Sentence Overlap Graph      │
│ • Okapi BM25 Formulation      │ │ • TextRank PageRank Algorithm │
│ • Levenshtein Fuzzy Search    │ │ • Definition Regex Extractor  │
│ • Highlighted <mark> Snippets │ │ • Interactive Exam Flashcards │
└───────────────────────────────┘ └───────────────────────────────┘
```

---

## 🧮 Mathematical Formulations

### 1. Sauvola Local Adaptive Thresholding
$$T(x, y) = m(x, y) \cdot \left[1 + k \cdot \left(\frac{s(x, y)}{R} - 1\right)\right]$$
Where $m(x, y)$ is the window local mean, $s(x, y)$ is the standard deviation, $R = 128$, and $k = 0.2$.

### 2. Okapi BM25 Document Scoring
$$\text{Score}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
Where $f(q_i, D)$ is term frequency, $|D|$ is document length, $\text{avgdl}$ is mean document length, $k_1 = 1.5$, and $b = 0.75$.

### 3. TextRank Sentence Graph Centrality
$$WS(V_i) = (1 - d) + d \cdot \sum_{V_j \in \text{In}(V_i)} \frac{\text{Weight}(V_j, V_i)}{\sum_{V_k \in \text{Out}(V_j)} \text{Weight}(V_j, V_k)} WS(V_j)$$
Where $d = 0.85$ is the PageRank damping factor and sentence similarity edge weights are computed via log-normalized token set intersections.

---

## 📁 Repository Structure
```text
handwritten-notes-to-searchable-notes/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated CI test suite & GitHub Pages deployer
├── app/
│   └── main.py                  # Full-featured Streamlit Web Dashboard
├── data/
│   └── sample_notes.json        # Curated engineering notes across 5 subjects
├── docs/
│   ├── index.html               # Live Client-Side PWA Web App (GitHub Pages)
│   └── VIVA_QUESTIONS_AND_ANSWERS.md # 25+ Comprehensive Academic Viva Q&A Guide
├── notebooks/
│   └── pipeline_walkthrough.ipynb # Jupyter research notebook with visual steps
├── src/
│   ├── __init__.py
│   ├── preprocessing.py         # Sauvola, deskew, line suppression, shadow removal
│   ├── ocr_engine.py            # Tesseract OCR wrapper + vision fallback
│   ├── text_cleaner.py          # Typo correction, hyphen unbreaker, Markdown formatter
│   ├── search_engine.py         # Inverted index, BM25 ranking & Levenshtein fuzzy search
│   └── summarizer.py            # TextRank PageRank summarizer & flashcard generator
├── tests/
│   ├── test_preprocessing.py    # Unit tests for computer vision filters
│   ├── test_ocr_engine.py       # Unit tests for OCR extraction & confidence
│   ├── test_text_cleaner.py     # Unit tests for typo correction & formatting
│   ├── test_search_engine.py    # Unit tests for BM25 ranking & fuzzy retrieval
│   ├── test_summarizer.py       # Unit tests for TextRank & flashcards
│   └── test_end_to_end_pipeline.py # Full system integration pipeline test
├── Dockerfile                   # Production Docker container definition
├── docker-compose.yml           # Compose specification
├── push_to_github.bat           # Deployment automation helper script
├── pytest.ini                   # Pytest configuration
├── requirements.txt             # Python dependencies
├── setup.py                     # Python package installer
├── LICENSE.md                   # MIT Open Source License
└── README.md                    # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Try Live in Browser (Zero Installation)
Visit the deployed web application on GitHub Pages:  
👉 **[https://khushiyadav35.github.io/handwritten-notes-to-searchable-notes/](https://khushiyadav35.github.io/handwritten-notes-to-searchable-notes/)**

### 2. Local Setup (Streamlit Dashboard)
```bash
# 1. Clone this repository
git clone https://github.com/khushiyadav35/handwritten-notes-to-searchable-notes.git
cd handwritten-notes-to-searchable-notes

# 2. Create virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Streamlit application
streamlit run app/main.py
```
Open your browser at `http://localhost:8501`.

### 3. Run with Docker
```bash
docker-compose up --build
```
Navigate to `http://localhost:8501`.

---

## 🧪 Automated Testing
The repository contains 29 comprehensive unit and integration tests:
```bash
pytest
```
```text
============================= 29 passed in 2.15s ==============================
```

---

## 🎓 Academic Defense & Viva Voce
Preparing for project presentation or external viva?  
Read the complete guide in [docs/VIVA_QUESTIONS_AND_ANSWERS.md](docs/VIVA_QUESTIONS_AND_ANSWERS.md) covering:
- Sauvola vs Otsu thresholding mathematical derivations
- Projection profile variance deskewing algorithm
- Inverted index vs Dense Vector Embeddings trade-offs
- TextRank PageRank graph formulation and convergence criteria
- Time & Space complexity analysis for every pipeline stage

---

## 👩‍💻 Author & Acknowledgements
**Khushi Yadav**  
B.Tech Computer Science & Engineering  
GitHub: [@khushiyadav35](https://github.com/khushiyadav35)
