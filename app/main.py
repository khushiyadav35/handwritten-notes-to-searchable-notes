"""
NoteScribe - Handwritten Notes to Searchable Digital Notes
Interactive Streamlit Application.
Author: Khushi Yadav (B.Tech CSE)
"""

import os
import json
import io
from datetime import datetime
import streamlit as st
from PIL import Image

# Import core modules
from src.preprocessing import preprocess_note_pipeline, load_image
from src.ocr_engine import NoteOCREngine
from src.text_cleaner import NoteTextCleaner
from src.search_engine import NoteSearchEngine, NoteDocument
from src.summarizer import NoteSummarizer

# Streamlit Page Config
st.set_page_config(
    page_title="NoteScribe | Handwritten Notes to Searchable Notes",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .badge-metric {
        background-color: #EEF2FF;
        color: #4F46E5;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .highlight-card {
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        background-color: #FFFFFF;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 14px;
    }
    .flashcard-box {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
        color: white;
        padding: 24px;
        border-radius: 12px;
        margin-top: 10px;
        min-height: 160px;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2);
    }
    mark {
        background-color: #FEF08A;
        padding: 1px 4px;
        border-radius: 3px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "search_engine" not in st.session_state:
    engine = NoteSearchEngine()
    # Pre-populate sample notes
    engine.add_document(NoteDocument(
        id="sample_os",
        title="Operating Systems: Process Scheduling",
        subject="Operating Systems",
        tags=["#exam", "#scheduling", "#cpu"],
        date_created=datetime.now().strftime("%Y-%m-%d"),
        confidence=94.2,
        content=(
            "## Operating Systems: Process Scheduling\n\n"
            "A Process is defined as a program in execution containing program counter, stack, and data section.\n"
            "- Process States: New, Ready, Running, Waiting, Terminated.\n"
            "- First-Come First-Served (FCFS): Non-preemptive scheduling with convoy effect.\n"
            "- Shortest Job First (SJF): Optimal algorithm minimizing average waiting time.\n"
            "- Round Robin (RR): Preemptive scheduling using a fixed time quantum.\n\n"
            "Formula: Turnaround Time = Completion Time - Arrival Time\n"
            "Formula: Waiting Time = Turnaround Time - Burst Time"
        )
    ))
    engine.add_document(NoteDocument(
        id="sample_ds",
        title="Data Structures: Binary Search Trees & AVL",
        subject="Data Structures",
        tags=["#trees", "#algorithms", "#balance"],
        date_created=datetime.now().strftime("%Y-%m-%d"),
        confidence=91.8,
        content=(
            "## Binary Search Trees (BST) & AVL Balancing\n\n"
            "A Binary Search Tree is a binary tree where left child < root < right child.\n"
            "- Inorder traversal of a BST always yields sorted order.\n"
            "- Average Search Time: O(log n), Worst Case: O(n) for skewed trees.\n"
            "- AVL Tree is defined as a self-balancing binary search tree with balance factor in {-1, 0, +1}.\n"
            "- Rotations: LL, RR (single rotations) and LR, RL (double rotations)."
        )
    ))
    st.session_state["search_engine"] = engine

if "ocr_engine" not in st.session_state:
    st.session_state["ocr_engine"] = NoteOCREngine()

if "text_cleaner" not in st.session_state:
    st.session_state["text_cleaner"] = NoteTextCleaner()

if "summarizer" not in st.session_state:
    st.session_state["summarizer"] = NoteSummarizer()

if "current_extracted_text" not in st.session_state:
    st.session_state["current_extracted_text"] = ""

if "current_cleaned_text" not in st.session_state:
    st.session_state["current_cleaned_text"] = ""


# Sidebar
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1517842645767-c639042777db?w=300&auto=format&fit=crop&q=80", use_container_width=True)
    st.title("NoteScribe AI")
    st.caption("Capstone Project | Khushi Yadav")
    st.markdown("---")
    st.markdown("### 📊 Notebook Stats")
    notes_count = len(st.session_state["search_engine"].documents)
    st.metric("Total Digital Notes", notes_count)
    
    tess_status = "✅ Active" if st.session_state["ocr_engine"].is_tesseract_installed() else "⚡ Fast Heuristic Vision"
    st.caption(f"**OCR Engine:** {tess_status}")
    st.caption("**Binarization:** Sauvola Adaptive")
    st.caption("**Ranking:** Okapi BM25 + Fuzzy")

    st.markdown("---")
    st.markdown("### 📥 Quick Export")
    json_export = st.session_state["search_engine"].to_json()
    st.download_button(
        label="Download Notebook Backup (JSON)",
        data=json_export,
        file_name="notescribe_backup.json",
        mime="application/json"
    )


# Header
st.markdown('<div class="main-header">📝 NoteScribe: Handwritten to Searchable Notes</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Convert messy photos of student notebooks into crystal-clear, searchable, summarized knowledge.</div>', unsafe_allow_html=True)

# Main Navigation Tabs
tab_scan, tab_search, tab_summary, tab_batch = st.tabs([
    "📸 Upload & Scan Note",
    "🔍 Digital Notebook & Search",
    "💡 AI Summary & Flashcards",
    "⚙️ Pipeline Benchmarks"
])


# ----------------------------------------------------
# TAB 1: UPLOAD & SCAN
# ----------------------------------------------------
with tab_scan:
    st.subheader("Step 1: Capture or Upload Handwritten Note")
    col_up, col_opts = st.columns([2, 1])

    with col_up:
        uploaded_file = st.file_uploader(
            "Upload image of handwritten notes (JPG, PNG, JPEG)",
            type=["jpg", "jpeg", "png"]
        )

    with col_opts:
        with st.expander("🛠️ Preprocessing Parameters", expanded=True):
            opt_lines = st.checkbox("Suppress Notebook Ruled Lines", value=True)
            opt_deskew = st.checkbox("Auto-Deskew & Tilt Correction", value=True)
            opt_bin_method = st.selectbox("Binarization Method", ["sauvola", "otsu"], index=0)

    if uploaded_file is not None:
        raw_image = Image.open(uploaded_file)

        with st.spinner("Processing image through vision pipeline..."):
            prep_res = preprocess_note_pipeline(
                raw_image,
                remove_paper_lines=opt_lines,
                auto_deskew=opt_deskew,
                binarize_method=opt_bin_method
            )

        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.image(prep_res["original"], caption="Original Capture", use_container_width=True)
        with col_img2:
            st.image(prep_res["pil_binary"], caption=f"Preprocessed (Binarized & Deskewed: {prep_res['skew_angle']}°)", use_container_width=True)

        # OCR trigger
        if st.button("🚀 Extract & Clean Handwritten Text", type="primary"):
            with st.spinner("Running handwriting recognition and contextual cleaner..."):
                ocr_out = st.session_state["ocr_engine"].extract_text(prep_res["pil_binary"])
                cleaned_out = st.session_state["text_cleaner"].clean(ocr_out.text)

                st.session_state["current_extracted_text"] = ocr_out.text
                st.session_state["current_cleaned_text"] = cleaned_out["clean_text"]
                st.session_state["current_ocr_conf"] = ocr_out.avg_confidence

    if st.session_state["current_cleaned_text"]:
        st.markdown("---")
        st.subheader("Step 2: Review, Edit & Save to Digital Notebook")

        conf = st.session_state.get("current_ocr_conf", 90.0)
        st.markdown(f'<span class="badge-metric">OCR Average Confidence: {conf}%</span>', unsafe_allow_html=True)

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.caption("Raw Extracted Text")
            st.text_area("Raw OCR", value=st.session_state["current_extracted_text"], height=250, key="raw_area")
        with col_t2:
            st.caption("Cleaned & Formatted Note (Editable)")
            edited_text = st.text_area("Cleaned Markdown", value=st.session_state["current_cleaned_text"], height=250, key="clean_area")

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            note_title = st.text_input("Note Title", value="Lecture Notes")
        with col_m2:
            note_subject = st.selectbox("Subject", [
                "Operating Systems",
                "Data Structures",
                "Computer Networks",
                "Database Systems",
                "Machine Learning",
                "General"
            ])
        with col_m3:
            note_tags = st.text_input("Tags (comma separated)", value="#exam, #lecture")

        if st.button("💾 Save to My Digital Notebook", type="primary"):
            tag_list = [t.strip() for t in note_tags.split(",") if t.strip()]
            new_id = f"note_{int(datetime.now().timestamp())}"
            doc = NoteDocument(
                id=new_id,
                title=note_title,
                content=edited_text,
                subject=note_subject,
                tags=tag_list,
                date_created=datetime.now().strftime("%Y-%m-%d"),
                confidence=conf
            )
            st.session_state["search_engine"].add_document(doc)
            st.success(f"✅ Note '{note_title}' saved and indexed successfully!")


# ----------------------------------------------------
# TAB 2: SEARCH & DIGITAL NOTEBOOK
# ----------------------------------------------------
with tab_search:
    st.subheader("Search Across All Handwritten Notes")

    col_q, col_s, col_t = st.columns([3, 1.5, 1.5])
    with col_q:
        search_query = st.text_input("🔍 Search keywords, concepts, or formulas (e.g. 'Dijkstra', 'Scheduling', 'BST')", "")
    with col_s:
        subject_filter = st.selectbox("Filter Subject", ["All", "Operating Systems", "Data Structures", "Computer Networks", "Database Systems", "Machine Learning"])
    with col_t:
        fuzzy_toggle = st.checkbox("Fuzzy Typo Match (OCR Tolerant)", value=True)

    results = st.session_state["search_engine"].search(
        query=search_query,
        subject_filter=None if subject_filter == "All" else subject_filter,
        enable_fuzzy=fuzzy_toggle
    )

    st.markdown(f"**Found {len(results)} matching notes:**")

    for res in results:
        doc = res.note
        with st.container():
            st.markdown(f"""
            <div class="highlight-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4 style="margin: 0; color: #1E293B;">{doc.title}</h4>
                    <span class="badge-metric">{doc.subject}</span>
                </div>
                <div style="color: #64748B; font-size: 0.85rem; margin-top: 4px;">
                    Date: {doc.date_created} | Tags: {', '.join(doc.tags)} | Match Score: {res.score}
                </div>
                <div style="margin-top: 10px; font-size: 0.95rem; color: #334155;">
                    <b>Snippet:</b> {res.snippet}
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("📖 View Full Note Content"):
                st.markdown(doc.content)
                col_exp1, col_exp2 = st.columns(2)
                with col_exp1:
                    st.download_button(
                        f"Download as Markdown (.md)",
                        data=doc.content,
                        file_name=f"{doc.id}.md",
                        mime="text/markdown",
                        key=f"dl_md_{doc.id}"
                    )
                with col_exp2:
                    if st.button(f"Generate AI Summary for this Note", key=f"sum_btn_{doc.id}"):
                        st.session_state["target_summary_text"] = doc.content
                        st.info("Switched to Tab 3 for summary!")


# ----------------------------------------------------
# TAB 3: AI SUMMARY & FLASHCARDS
# ----------------------------------------------------
with tab_summary:
    st.subheader("Automatic Extractive Summarization & Flashcards")
    st.write("Generate concise exam revision sheets and self-study question cards instantly.")

    default_text = st.session_state.get(
        "target_summary_text",
        st.session_state["search_engine"].documents.get("sample_os", NoteDocument("", "", "")).content
    )

    text_to_summarize = st.text_area("Note Text to Summarize", value=default_text, height=180)
    col_sum_opt, col_sum_btn = st.columns([2, 1])

    with col_sum_opt:
        num_sents = st.slider("Target Summary Sentences", 1, 5, 3)

    if st.button("✨ Generate Summary & Flashcards", type="primary"):
        with st.spinner("Analyzing sentence graphs with TextRank..."):
            summary_res = st.session_state["summarizer"].summarize(text_to_summarize, max_summary_sentences=num_sents)

            st.markdown("### 📌 Executive TL;DR")
            st.info(summary_res.executive_summary)
            st.caption(f"Word Reduction: {summary_res.compression_ratio}% shorter than original note.")

            col_pts, col_defs = st.columns(2)
            with col_pts:
                st.markdown("### 🎯 High-Yield Key Points")
                for pt in summary_res.key_points:
                    st.markdown(f"- {pt}")

            with col_defs:
                st.markdown("### 📖 Extracted Definitions & Formulas")
                if summary_res.definitions:
                    for term, val in summary_res.definitions.items():
                        st.markdown(f"**{term}**: {val}")
                else:
                    st.caption("No explicit definition patterns detected.")

            st.markdown("---")
            st.markdown("### 🃏 Interactive Exam Flashcards")
            if summary_res.flashcards:
                for idx, card in enumerate(summary_res.flashcards):
                    with st.expander(f"Card {idx+1}: {card.question}", expanded=(idx == 0)):
                        st.markdown(f"**Answer:** {card.answer}")
            else:
                st.caption("Add more content to automatically generate flashcards.")


# ----------------------------------------------------
# TAB 4: BENCHMARKS & ARCHITECTURE
# ----------------------------------------------------
with tab_batch:
    st.subheader("System Performance & Academic Evaluation")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Pipeline Stage 1", "Preprocessing", "Sauvola + Deskew")
    col_m2.metric("Pipeline Stage 2", "OCR Extraction", "LSTM Engine")
    col_m3.metric("Pipeline Stage 3", "Search Model", "Okapi BM25")
    col_m4.metric("Pipeline Stage 4", "Summarizer", "TextRank PageRank")

    st.markdown("### 🔬 Architectural Flow")
    st.code("""
    1. Input: Smartphone Camera Capture (JPG/PNG)
    2. Shadow Removal: Gaussian background division
    3. Rule Line Suppression: Horizontal 1x31 morphological opening
    4. Deskewing: Projection profile variance optimization
    5. Binarization: Local adaptive Sauvola thresholding (k=0.2, window=25)
    6. OCR: Tesseract LSTM with PSM 6 (single uniform block)
    7. Post-Processing: Hyphen unbreaking, OCR confusion correction
    8. Retrieval: Inverted index with BM25 (k1=1.5, b=0.75) + Levenshtein distance
    9. Summarization: Graph-based TextRank PageRank with salience damping
    """, language="text")
