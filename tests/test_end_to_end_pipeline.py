"""
End-to-End System Integration Test for NoteScribe Pipeline.
"""

from PIL import Image, ImageDraw
from src.preprocessing import preprocess_note_pipeline
from src.ocr_engine import NoteOCREngine
from src.text_cleaner import NoteTextCleaner
from src.search_engine import NoteSearchEngine, NoteDocument
from src.summarizer import NoteSummarizer


def test_full_student_note_workflow():
    # 1. Create a simulated student notebook photo with lines and text
    width, height = 450, 250
    img = Image.new("RGB", (width, height), color=(240, 240, 235))
    draw = ImageDraw.Draw(img)
    for y in range(40, height, 40):
        draw.line([(10, y), (width - 10, y)], fill=(190, 210, 225), width=1)
    draw.text((25, 25), "LECTURE: COMPUTER NETWORKS", fill=(10, 10, 40))

    # 2. Image Preprocessing
    prep_res = preprocess_note_pipeline(img, remove_paper_lines=True, auto_deskew=True)
    assert prep_res["binary"].shape == (height, width)

    # 3. OCR Extraction
    lecture_transcription = (
        "LECTURE: COMPUTER NETWORKS\n"
        "The Open Systems Interconnection (OSI) model defines a 7-layer networking frame-\n"
        "work. Layer 7 is the Application layer, providing network services directly to end users.\n"
        "Layer 4 is the Transport layer (TCP and UDP) ensuring reliable end-to-end communication.\n"
        "Formula: Bandwidth-Delay Product = Bandwidth * Round Trip Time.\n"
    )
    ocr_engine = NoteOCREngine()
    ocr_res = ocr_engine.extract_text(prep_res["pil_binary"], fallback_text=lecture_transcription)
    assert ocr_res.avg_confidence > 75.0

    # 4. Text Cleaning
    cleaner = NoteTextCleaner()
    cleaned = cleaner.clean(ocr_res.text)
    clean_text = cleaned["clean_text"]
    assert "framework" in clean_text
    assert "frame-" not in clean_text

    # 5. Search Indexing
    search_engine = NoteSearchEngine()
    doc = NoteDocument(
        id="cn_lec_1",
        title="OSI Model & Transport Protocols",
        content=clean_text,
        subject="Computer Networks",
        tags=["#osi", "#networking", "#exam"]
    )
    search_engine.add_document(doc)

    # Search for term
    query_res = search_engine.search("Transport layer", subject_filter="Computer Networks")
    assert len(query_res) == 1
    assert query_res[0].note.id == "cn_lec_1"
    assert "<mark>" in query_res[0].snippet

    # 6. Extractive Summarizer
    summarizer = NoteSummarizer()
    sum_res = summarizer.summarize(clean_text, max_summary_sentences=2)
    assert len(sum_res.key_points) > 0
    assert len(sum_res.flashcards) > 0
