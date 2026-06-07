# India Runs Hackathon: Track 1 - Intelligent Candidate Discovery & Ranking

A production-grade, highly optimized multi-stage search and ranking pipeline designed to discover the top 100 best-fit candidates from a pool of 100,000 resumes for a **Senior AI Engineer** role at Redrob.

---

## 🎯 Executive Summary
Finding a Senior AI Engineer in Pune/Noida with 5–9 years of experience and deep expertise in embedding, retrieval, and ranking requires more than simple keyword matching or slow, expensive LLM-based parsing. 

This solution uses a **3-stage hybrid search pipeline** (Lexical Search + Dense Semantic Reranking + Behavioral & Exclusions Signals) fused using **Reciprocal Rank Fusion (RRF)** to deliver a highly accurate candidate ranking. It is optimized to run locally on standard hardware without external API dependencies or GPU acceleration.

---

## ⚡ Hard Constraints & Compliance
The solution strictly adheres to the hackathon's resource constraints:
- **Runtime:** **~3.07 minutes** wall-clock (well under the **5-minute limit**).
- **Compute:** 100% **CPU-only**; no GPU utilized.
- **Memory:** Max peak RAM is **well below 16 GB** (utilizes streaming loaders and selective candidate subsampling).
- **Network:** **100% offline** during execution; no calls to external LLMs (Groq, OpenAI, Anthropic, Cohere, etc.).
- **Disk Space:** Output and intermediate files consume less than **50 MB** (limit is 5 GB).
- **Validation:** 100% compliant with the challenge's schema validator (verified via `validate_submission.py`).

---

## 🏗️ Pipeline Architecture

The pipeline consists of three main stages, followed by reciprocal rank fusion and non-templated reasoning generation.

```
[100,000 Candidates]
         │
         ▼
┌──────────────────────────────────────────────┐
│  Stage 1: Lexical Search (BM25)              │  <-- Filter 100K -> 2K candidates
└──────────────────────────────────────────────┘
         │ (Top 2,000 candidates)
         ▼
┌──────────────────────────────────────────────┐
│  Stage 2: Dense Semantic Reranking           │  <-- Rerank using all-MiniLM-L6-v2
└──────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│  Stage 3: Behavioral & Exclusion Signals     │  <-- Score YoE, Git activity, Honeypots
└──────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│  RRF Fusion & Format-Aligned Tie-Breaking    │  <-- Combine ranks & sort ties alphabetically
└──────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│  Reasoning Generation & CSV Write            │  <-- Write final submission.csv
└──────────────────────────────────────────────┘
```

### 1. Stage 1: Lexical Search (BM25)
* **Goal:** High-recall filtering to reduce the candidate pool from 100,000 to the top 2,000.
* **Text Representation:** Structured candidate profile concatenations including current job title, skills, summary, and recent work history.
* **Query Optimization:** Curated query constructed from the job description (JD) prepended with weighted terms like `embeddings`, `retrieval`, `ranking`, `FAISS`, `RAG`, etc., while excluding "consulting traps" and negative-match words (e.g. `tcs`, `wipro`, `accenture`, etc.) that degrade retrieval relevance.

### 2. Stage 2: Dense Semantic Reranking (Sentence Transformers)
* **Goal:** High-precision semantic matching between the job description and the top 2,000 candidates.
* **Model:** `all-MiniLM-L6-v2` (run locally on CPU), which maps sentences and paragraphs into a 384-dimensional dense vector space.
* **Scoring:** Computes Cosine Similarity between the JD embedding and each of the 2,000 candidate text embeddings.

### 3. Stage 3: Behavioral & Exclusions Signals
* **Goal:** Adjust scores based on candidate location, experience depth, availability, and screen out "honeypots" or mismatched roles.
* **Target Fit Scoring:**
  - **Experience Matching:** Ideal score for 5–9 years of experience. Candidates below 5 years or above 9 years are progressively penalized but not ignored.
  - **Recruiter Response Rate:** Direct positive scalar for high responsiveness (availability signal).
  - **GitHub Contributor Score:** Weighted boost for active open-source contributions.
  - **Notice Period & Availability:** Upward boost for `open_to_work` and short notice periods (e.g., 0-30 days); penalties applied for extremely long notice periods (e.g., 90-120 days).
* **Honeypots & Noise Exclusion:**
  - Strict penalty checks for non-technical candidates (e.g., HR, Sales, Content) containing matching keyword traps.
  - Verification of foundational AI/ML skills to prevent generic Software Engineers from ranking high.

### 4. Reciprocal Rank Fusion (RRF) & Tie-Breaking
* RRF combines semantic rank and behavioral rank into a unified score using the formula:
  $$\text{RRF Score}_i = \sum_{m \in M} \frac{1}{60 + \text{Rank}_{i, m}}$$
* **Exact Float Tie-Breaking:** To prevent rounding conflicts with the validator script, the final list is sorted using `(-float(f"{rrf_score:.6f}"), candidate_id)`. This ensures that any identical rounded scores in the CSV are perfectly ordered alphabetically by `candidate_id`.

---

## 📂 Project Structure

```
├── main.py                     # Entry point orchestrating the pipeline stages
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── architecture_presentation.pdf # Summary slide-deck of architecture & results
├── ranker/                     # Core python module
│   ├── __init__.py             # Module initialization
│   ├── loader.py               # Data parsing, text representations & IO helpers
│   ├── bm25_stage.py           # Stage 1: BM25 index & lexical retrieval
│   ├── embed_stage.py          # Stage 2: Local SentenceTransformer embed & cosine similarity
│   ├── signals.py              # Stage 3: Behavioral scoring & Honeypot checking
│   ├── fusion.py               # Rank fusion (RRF) and formatting-aligned tie-breaking
│   └── reasoning.py            # Automated reasoning generator for final report
└── submission.csv              # Generated final outputs (Top 100 ranked candidates)
```

---

## 🚀 Installation & Usage

### 1. Prerequisites
- Python 3.9 or higher
- PyTorch (CPU version recommended for faster startup and footprint)

### 2. Setup
Clone the repository, initialize your virtual environment, and install dependencies:
```bash
# Clone the repository
git clone https://github.com/your-username/india-runs-ranker.git
cd india-runs-ranker

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Execution
Specify the correct paths in `main.py` if your raw data folder structure differs, and run:
```bash
python main.py
```
This will output the `submission.csv` containing columns `candidate_id`, `rank`, `score`, and `reasoning`.

### 4. Verification
Run the validation script to verify formatting compliance:
```bash
python validate_submission.py submission.csv
```

---

## 📊 Performance & Accuracy Highlights
- **BM25 Retrieval:** Reduces search space from 100,000 to 2,000 in **~48 seconds**.
- **Dense Embedding Encoding:** Encodes the 2,000 candidate texts and JD in **~129 seconds** on CPU.
- **RRF & Filtering:** Fuses rankings and resolves ties in **~0.01 seconds**.
- **Validator Compliance:** Passed with **0 errors**. Ranks 87 and 88, which both rounded to a score of `0.011188`, are correctly ordered alphabetically (`CAND_0006418` before `CAND_0076831`).
