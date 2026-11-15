# NoteScribe - Comprehensive Engineering Viva Voce & Defense Guide

**Project Title:** NoteScribe: Handwritten Notes to Searchable Digital Notes  
**Author:** Khushi Yadav (B.Tech Computer Science & Engineering)  
**Domain:** Computer Vision, Optical Character Recognition (OCR), Information Retrieval, Natural Language Processing (NLP)

---

## 🏛️ Category 1: Computer Vision & Image Preprocessing

### Q1. Why is direct OCR on phone camera photos of handwritten notes prone to failure?
**Answer:**  
Standard smartphone captures of notebooks face five core degradations:
1. **Non-Uniform Illumination & Shadows:** Overhead room lighting or the student's hand casts ambient shadows, causing standard global binarization (like simple 128 threshold) to clip text into solid black blobs.
2. **Notebook Ruled Lines:** Blue or black horizontal ruling intersects characters; OCR models mistake ruled lines for hyphens (`-`), underscores (`_`), or minus signs.
3. **Perspective Skew & Tilt:** Handheld photography introduces rotational tilt ($\pm 15^\circ$) and perspective keystoning.
4. **Paper Grain & Ink Degradation:** Ballpoint and pencil strokes have variable stroke thickness and low contrast relative to off-white paper.
5. **Crumpled Surfaces & Binding Curves:** Page curvature causes nonlinear line bending near notebook spirals.

### Q2. How does Sauvola adaptive thresholding differ from Otsu thresholding?
**Answer:**  
- **Otsu's Thresholding:** Computes a single global threshold $T$ that minimizes inter-class variance between foreground and background over the entire image histogram. It fails completely when an image has a shadow across one half, because a single threshold cannot separate text from dark background.
- **Sauvola's Thresholding:** Computes a local threshold $T(x, y)$ for each pixel using a local window ($W \times W$):
  $$T(x, y) = m(x, y) \cdot \left[1 + k \cdot \left(\frac{s(x, y)}{R} - 1\right)\right]$$
  Where:
  - $m(x, y)$ is the local window mean.
  - $s(x, y)$ is the local window standard deviation.
  - $R$ is the maximum standard deviation ($R = 128$ for 8-bit images).
  - $k$ is a tuning parameter ($k \approx 0.2$).
  Sauvola dynamically adapts to local contrast, preserving faint pencil strokes even in shadowy regions.

### Q3. How does NoteScribe suppress horizontal notebook ruled lines without destroying characters?
**Answer:**  
We employ mathematical morphology:
1. Invert the grayscale image so text and ruled lines become foreground ($> 0$).
2. Apply a morphological opening with a wide horizontal 1D structuring element $K = \text{ones}(1, 31)$. Because notebook lines are continuous horizontally while letter strokes are vertical or diagonal, opening eliminates vertical strokes and retains the ruled lines.
3. Dilate the detected line mask slightly ($2 \times 2$) to capture anti-aliasing edges.
4. Inpaint or replace pixels under the line mask with the surrounding background intensity ($220-255$).

### Q4. How is deskewing calculated mathematically?
**Answer:**  
We use the **Horizontal Projection Profile Variance Method**:
1. For an angle $\theta \in [-20^\circ, +20^\circ]$ with step size $0.5^\circ$, rotate the foreground binary array.
2. Compute the horizontal projection profile $P_\theta(y) = \sum_{x} I_\theta(x, y)$.
3. Compute the variance of $P_\theta$:
   $$\text{Var}(P_\theta) = \frac{1}{H} \sum_{y=1}^{H} (P_\theta(y) - \mu)^2$$
4. When text lines are oriented horizontally, text peaks are sharply defined and inter-line spaces are zero, maximizing variance. The angle $\theta^*$ yielding maximum variance is chosen as the true skew angle.

---

## 🔍 Category 2: OCR & Post-Processing Cleaning

### Q5. What is the internal architecture of modern Tesseract 4/5?
**Answer:**  
Tesseract 4 and 5 utilize an end-to-end Neural Network architecture:
1. **Feature Extraction:** Convolutional layers extract visual feature maps from normalized line image slices.
2. **Sequence Modeling:** Bidirectional Long Short-Term Memory (BiLSTM) networks model horizontal dependencies forward and backward across character sequences.
3. **CTC Loss (Connectionist Temporal Classification):** Enables training on unaligned text lines without needing individual character bounding box segmentations.
4. **Language Model Decoding:** Beam search decoding incorporates n-gram word dictionaries and character frequencies.

### Q6. Why is a deterministic text cleaner needed after OCR?
**Answer:**  
Handwriting has high character confusion entropy:
1. **Broken Hyphenated Line-wraps:** Words at line endings like `concur-` \n `rency` must be reassembled into `concurrency` for search indexers to match queries.
2. **Character Substitutions:** Digits frequently swap into alphabetic tokens (`pr0cess` $\to$ `process`, `syst3m` $\to$ `system`, `1` $\to$ `l` or `I`).
3. **Margin Artifacts:** Binding rings and notebook margins produce isolated dots and pipe (`|`) noise.
Our `NoteTextCleaner` applies regex heuristics and word-boundary substitutions to restore proper grammatical structure and Markdown headings.

---

## 🗄️ Category 3: Information Retrieval & Search

### Q7. Explain the mathematical formulation of Okapi BM25 used in NoteScribe.
**Answer:**  
Given a query $Q$ with tokens $q_1, q_2, \dots, q_n$ and document $D$:
$$\text{Score}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
Where:
- $f(q_i, D)$ is term frequency of token $q_i$ in document $D$.
- $|D|$ is the length of document $D$ in words.
- $\text{avgdl}$ is the average document length across the entire notebook index.
- $k_1$ (set to $1.5$) calibrates term frequency saturation.
- $b$ (set to $0.75$) controls length normalization penalty.
- $\text{IDF}(q_i) = \ln\left(1 + \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5}\right)$ gives higher weight to rare, informative technical terms.

### Q8. Why is Levenshtein fuzzy search essential for handwritten notes?
**Answer:**  
Even with high-accuracy OCR, handwritten words frequently have single-character OCR misrecognitions (e.g. `dijkstr4` for `dijkstra`, `av1` for `avl`). If search relied solely on exact string equality, the student would never find the note! By expanding queries to vocabulary tokens within Levenshtein edit distance $\le 1$, NoteScribe provides fault-tolerant recall.

---

## 🧠 Category 4: NLP & Extractive Summarization

### Q9. How does TextRank work for note summarization?
**Answer:**  
TextRank models text as an undirected graph $G = (V, E)$:
- **Vertices ($V$):** Each sentence in the note is a node.
- **Edges ($E$):** The weight of an edge between sentence $S_i$ and $S_j$ is their lexical content overlap:
  $$\text{Weight}(S_i, S_j) = \frac{|\text{Tokens}(S_i) \cap \text{Tokens}(S_j)|}{\log(|\text{Tokens}(S_i)|) + \log(|\text{Tokens}(S_j)|)}$$
- **PageRank Formulation:**
  $$WS(V_i) = (1 - d) + d \cdot \sum_{V_j \in \text{In}(V_i)} \frac{\text{Weight}(V_j, V_i)}{\sum_{V_k \in \text{Out}(V_j)} \text{Weight}(V_j, V_k)} WS(V_j)$$
  Where $d = 0.85$ is the damping factor.
Sentences with high central connectivity represent the core thesis of the note and are extracted as the executive summary.

### Q10. What is the difference between Extractive and Abstractive summarization, and why chose Extractive for study notes?
**Answer:**  
- **Extractive Summarization:** Selects exact, high-salience sentences and definitions directly from the student's text.
- **Abstractive Summarization:** Paraphrases and generates novel sentences using large sequence-to-sequence models (e.g. T5, BART).
- **Why Extractive for Notes:**
  1. **Zero Hallucination:** Academic formulas, numerical theorems, and definitions must not be altered or distorted by LLM hallucination.
  2. **Zero Compute Overhead:** Runs locally and instantaneously in browser or serverless environments without requiring expensive GPU inference servers.
  3. **Privacy & Offline Access:** Student personal study material stays local on device.

---

## 💻 Category 5: Engineering & Scalability

### Q11. What is the time and space complexity of the pipeline?
- **Preprocessing:** $\mathcal{O}(W \cdot H)$ pixel operations. Sauvola with integral images runs in linear time.
- **OCR:** $\mathcal{O}(L \cdot W_{\text{line}})$ where $L$ is number of text lines.
- **BM25 Search:** $\mathcal{O}(|Q| \cdot \text{PostingListLength})$, yielding sub-millisecond retrieval across thousands of notes.
- **TextRank:** $\mathcal{O}(S^2)$ where $S$ is number of sentences per note (typically $10-50$, executing in $< 5\text{ms}$).

### Q12. What are the limitations and future enhancements?
1. **Mathematical Equations & Diagrams:** Future work can integrate LaTeX OCR models (such as Nougat or TrOCR) and diagram vectorizers.
2. **Cursive Ligature Disambiguation:** Fine-tuning a Vision Transformer (ViT) on Indic and English student handwriting datasets (e.g. IAM Handwriting Database).
3. **Cross-Device Sync:** WebRTC / P2P synchronization across smartphone camera and laptop web client.
