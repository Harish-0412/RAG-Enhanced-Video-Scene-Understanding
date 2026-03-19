# VideoSceneRAG: AI-Powered Video Scene Understanding

An advanced Retrieval-Augmented Generation (RAG) system that transforms video content into an intelligent, queryable knowledge base. Ask natural language questions about any indexed video and receive precise, timestamped answers grounded in visual and audio content.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [System Design](#system-design)
- [Data Flow](#data-flow)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [License](#license)

---

## 🎯 Overview

VideoSceneRAG is a multimodal AI system that leverages computer vision, speech recognition, and large language models to understand video content at a semantic level. It enables users to:

- **Ask questions** about video content using natural language
- **Get timestamped answers** pinpointed to specific video scenes
- **Retrieve clips** based on conceptual understanding, not just keyword matching
- **Maintain conversation context** across multiple turns
- **Verify hallucinations** by grounding answers in actual video content

The system processes videos through a 4-phase pipeline that combines scene detection, audio transcription, multimodal embeddings, and LLM-based reasoning.

---

## ✨ Key Features

### Phase 1: Intelligent Frame Extraction
- **Temporal Segmentation**: Content-aware scene detection via PySceneDetect
- **Adaptive Bitrate Downsampling**: Reduces processing overhead for large videos
- **Visual Deduplication**: Perceptual hashing eliminates near-duplicate frames
- **Center-Frame Strategy**: Captures representative frame from each scene

### Phase 2: Multimodal Content Analysis
- **Automatic Speech Recognition**: Faster-Whisper extracts transcripts with word-level timestamps
- **Visual Feature Extraction**: Deep learning models generate visual descriptions
- **Temporal Alignment**: Audio and visual features aligned to scene boundaries
- **Confidence Scoring**: Flags low-confidence transcriptions

### Phase 3: Hybrid Retrieval Indexing
- **Dual-Collection Strategy**: 
  - Dense embeddings for semantic similarity (SentenceTransformer)
  - Sparse collection for keyword/OCR text matching
- **ChromaDB Integration**: Persistent vector storage with metadata preservation
- **Reciprocal Rank Fusion**: Combines dense + sparse results intelligently
- **Temporal Re-ranking**: Prioritizes contextually nearby scenes

### Phase 4: NLP-Powered RAG Engine
- **Intent Classification**: Routes queries to optimal retrieval strategy
  - CONCEPT, TIMESTAMP, COMPARE, SUMMARISE, FOLLOWUP modes
- **Query Rewriting**: LLM semantically expands user questions for better recall
- **Grounded Answer Generation**: Gemini generates answers with inline citations
- **Conversation Memory**: Multi-turn context awareness
- **Hallucination Detection**: Validates answer factuality against source material
- **Confidence Scoring**: Composite score from retrieval quality + grounding checks

---

## 🏗️ Architecture

### High-Level System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                         VideoSceneRAG System                        │
└────────────────────────────────────────────────────────────────────┘

                            INPUT: Video File
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
            ┌─────────────────┐          ┌──────────────────┐
            │ PHASE 1: Frame  │          │ PHASE 1: Frame   │
            │ Extraction &    │          │ Deduplication    │
            │ Temporal Seg.   │          │ (pHash)          │
            └────────┬────────┘          └────────┬─────────┘
                     │                            │
                     └────────────────┬───────────┘
                                      ▼
                    ┌──────────────────────────────┐
                    │ Metadata: mapping.json       │
                    │ (timestamps, scenes, frames) │
                    └────────────────┬─────────────┘
                                     │
                ┌────────────────────┼────────────────────┐
                ▼                    ▼                    ▼
         ┌───────────────┐   ┌──────────────┐   ┌──────────────┐
         │ PHASE 2A:     │   │ PHASE 2B:    │   │ PHASE 2B:    │
         │ ASR (Whisper) │   │ Visual       │   │ OCR/Text     │
         │ Transcription │   │ Description  │   │ Extraction   │
         └───────┬───────┘   └──────┬───────┘   └──────┬───────┘
                 │                  │                  │
                 └──────────────────┬──────────────────┘
                                    ▼
                    ┌──────────────────────────────┐
                    │ enriched_metadata.json:      │
                    │ • combined_context (text)    │
                    │ • scene_transcript           │
                    │ • visual_description         │
                    │ • on_screen_text             │
                    │ • timestamps & confidence    │
                    └────────────────┬─────────────┘
                                     │
            ┌────────────────────────┼────────────────────────┐
            ▼                        ▼                        ▼
      ┌──────────────┐        ┌──────────────┐        ┌──────────────┐
      │ PHASE 3:     │        │ PHASE 3:     │        │ PHASE 3:     │
      │ Dense        │        │ Sparse       │        │ Metadata     │
      │ Embedding    │        │ Embedding    │        │ Normalization│
      │ (STEncoder)  │        │ (Keywords)   │        │              │
      └──────┬───────┘        └──────┬───────┘        └──────┬───────┘
             │                       │                       │
             └───────────────────────┼───────────────────────┘
                                     ▼
                    ┌──────────────────────────────┐
                    │ ChromaDB Collections:        │
                    │ • video_moments_dense        │
                    │ • video_moments_sparse       │
                    └────────────────┬─────────────┘
                                     │
                                     ▼
                    ┌──────────────────────────────┐
                    │ PHASE 4: RAG Query Engine    │
                    │                              │
                    │ • Intent Classification      │
                    │ • Query Rewriting            │
                    │ • Hybrid Retrieval (RRF)     │
                    │ • Temporal Re-ranking        │
                    │ • Grounded Answer Gen        │
                    │ • Hallucination Detection    │
                    │ • Confidence Scoring         │
                    └────────────────┬─────────────┘
                                     │
                                     ▼
                    ┌──────────────────────────────┐
                    │ OUTPUT: Timestamped Answer   │
                    │ with Citations & Confidence  │
                    │ [Scene @ HH:MM:SS]           │
                    └──────────────────────────────┘
```

### Component Interaction Diagram

```
                    ┌─────────────────┐
                    │   main.py       │
                    │   (CLI REPL)    │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
    ┌─────────────────┐ ┌────────────┐ ┌─────────────┐
    │ phase1          │ │ phase2     │ │ phase3      │
    │ _sampling       │ │ _audio.py  │ │ _indexing   │
    │ .py             │ │ _visual.py │ │ .py         │
    └────────┬────────┘ └────────┬───┘ └──────┬──────┘
             │                   │             │
             ▼                   ▼             ▼
        ┌─────────┐         ┌──────────┐  ┌──────────┐
        │ OpenCV  │         │ Whisper  │  │ ChromaDB │
        │ Scene   │         │ SentTF   │  │ Chroma   │
        │ Detect  │         │ Models   │  │ Client   │
        └────┬────┘         └────┬─────┘  └────┬─────┘
             │                   │             │
             └───────────────────┼─────────────┘
                                 │
                                 ▼
                    ┌─────────────────────┐
                    │ VideoRetriever      │
                    │ (phase3_indexing)   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ phase4_rag.py       │
                    │ • VideoRAG class    │
                    │ • ConvMemory        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Gemini API          │
                    │ (google.generativeai)│
                    └─────────────────────┘
```

---

## 🔧 System Design

### Data Model

#### Frame & Scene Metadata (Phase 1)
```json
{
  "scene_id": "scene_001",
  "start_frame": 0,
  "end_frame": 120,
  "start_seconds": 0.0,
  "end_seconds": 4.8,
  "center_frame_path": "frames/scene_001_center.jpg",
  "frame_count": 121,
  "avg_brightness": 127.5,
  "motion_intensity": 0.45,
  "visual_hash": "abcd1234..."
}
```

#### Enriched Scene Record (Phase 2 → 3)
```json
{
  "scene_id": "scene_001",
  "timestamp": 0.0,
  "duration": 4.8,
  "combined_context": "A person is typing on a laptop...",
  "scene_transcript": "Let me show you the code...",
  "transcript_confidence": 0.95,
  "visual_description": "Indoor office setting, bright lighting...",
  "on_screen_text": ["Python", "def main()"],
  "word_count": 15,
  "frame_path": "frames/scene_001_center.jpg"
}
```

#### ChromaDB Collection Structure
```
Collection: video_moments_dense
├── id: 'scene_001'
├── embedding: [0.12, -0.45, 0.78, ...]  (384-dim SentenceTransformer)
├── metadata: {all fields from enriched record}
└── document: combined_context

Collection: video_moments_sparse
├── id: 'scene_001_sparse'
├── embedding: [keyword TF-IDF sparse vector]
├── metadata: {same as above}
└── document: on_screen_text + visual description
```

### Retrieval Strategy

#### Intent-Driven Routing

| Intent | Strategy | Example |
|--------|----------|---------|
| **CONCEPT** | Hybrid Dense + Sparse (RRF) | "Find scenes with coding" |
| **TIMESTAMP** | Temporal window lookup | "What happens at 2 minutes?" |
| **COMPARE** | Dual sub-queries merged | "Compare scene A vs scene B" |
| **SUMMARISE** | Broad top-K (k=15-20) | "Summarize the video" |
| **FOLLOWUP** | Prior context + narrow query | "And then what?" |

#### Hybrid Retrieval (RRF) Algorithm

```
Dense Results:    [scene_1(rank=1), scene_5(rank=2), scene_3(rank=3), ...]
Sparse Results:   [scene_3(rank=1), scene_1(rank=2), scene_7(rank=3), ...]

RRF Score Calculation:
  score(d) = Σ 1 / (k + rank(d, i)) for each collection i
  
  RRF Rank 1: scene_1  = 1/(0.5+1) + 1/(0.5+2) = 0.67 + 0.40 = 1.07
  RRF Rank 2: scene_3  = 1/(0.5+3) + 1/(0.5+1) = 0.25 + 0.67 = 0.92
  RRF Rank 3: scene_5  = 1/(0.5+2) + 0            = 0.40 + 0    = 0.40
  RRF Rank 4: scene_7  = 0 + 1/(0.5+3)            = 0 + 0.25    = 0.25

Final Ranked: [scene_1, scene_3, scene_5, scene_7, ...]
```

### Confidence Scoring

```
confidence_score = (retrieval_score * α) + (grounding_ratio * β) + (relevance_boost * γ)

where:
  α, β, γ = tunable weights (typically 0.3, 0.5, 0.2)
  retrieval_score = mean cosine similarity of top-k docs
  grounding_ratio = (supported_claims / total_claims)
  relevance_boost = 1.0 if high temporal coherence, else 0.8
```

---

## 📊 Data Flow

### Complete Pipeline Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                      VIDEO INPUT (MP4)                           │
└────────────────────────┬─────────────────────────────────────────┘
                         │
     ┌───────────────────▼───────────────────┐
     │                                       │
     │     PHASE 1: SCENE DETECTION          │
     │     ─────────────────────────         │
     │  1. Load video with OpenCV            │
     │  2. PySceneDetect identifies cuts     │
     │  3. Extract center frame per scene    │
     │  4. Visual deduplication (pHash)      │
     │  5. Compute motion metrics (SAD)      │
     │                                       │
     │  OUTPUT: mapping.json                 │
     │  ├─ scene_id, timestamps              │
     │  ├─ center_frame_path                 │
     │  └─ motion scores                     │
     └───────────────────┬───────────────────┘
                         │
     ┌───────────────────▼─────────────────────────────────┐
     │                                                     │
     │           PHASE 2A: ASR & TRANSCRIPTION             │
     │           ──────────────────────────                │
     │  1. Extract audio from video (MoviePy)             │
     │  2. Faster-Whisper ASR on full audio track        │
     │  3. Word-level timestamps & confidence             │
     │  4. Align segments to scenes (temporal overlap)    │
     │  5. Aggregate to scene-level transcripts           │
     │                                                     │
     │  OUTPUT: transcript.json (all ASR segments)        │
     └───────────────────┬─────────────────────────────────┘
                         │
     ┌───────────────────▼──────────────────────────────────┐
     │                                                      │
     │          PHASE 2B: VISUAL & TEXT ANALYSIS            │
     │          ──────────────────────────────              │
     │  1. Load center frames from mapping.json            │
     │  2. Vision model generates descriptions             │
     │  3. OCR extracts on-screen text                     │
     │  4. Build combined_context (text + audio + visual)  │
     │  5. Confidence scoring (transcript_confidence)      │
     │                                                      │
     │  OUTPUT: enriched_metadata.json                      │
     │  ├─ scene_id, timestamp, duration                   │
     │  ├─ combined_context (multimodal text)              │
     │  ├─ scene_transcript                                │
     │  ├─ visual_description                              │
     │  ├─ on_screen_text                                  │
     │  └─ word_count, confidence                          │
     └───────────────────┬──────────────────────────────────┘
                         │
     ┌───────────────────▼──────────────────────────────────┐
     │                                                      │
     │      PHASE 3A: EMBEDDING & INDEXING (Dense)         │
     │      ────────────────────────────────────            │
     │  1. Load enriched_metadata.json                      │
     │  2. Initialize SentenceTransformer encoder           │
     │  3. Embed combined_context → 384-dim vectors        │
     │  4. Create ChromaDB collection (video_moments_dense)│
     │  5. Upsert all scenes with metadata                 │
     │                                                      │
     │  ChromaDB Schema (Dense):                           │
     │  ├─ id: scene_id                                    │
     │  ├─ embedding: [384 dims]                           │
     │  ├─ metadata: {all enriched fields}                 │
     │  └─ document: combined_context                      │
     └───────────────────┬──────────────────────────────────┘
                         │
     ┌───────────────────▼──────────────────────────────────┐
     │                                                      │
     │      PHASE 3B: EMBEDDING & INDEXING (Sparse)        │
     │      ────────────────────────────────────            │
     │  1. Extract keywords from on_screen_text + visual   │
     │  2. Build TF-IDF sparse vectors                     │
     │  3. Create ChromaDB collection (video_moments_sparse)
     │  4. Upsert keyword-based documents                  │
     │                                                      │
     │  ChromaDB Schema (Sparse):                          │
     │  ├─ id: scene_id_sparse                             │
     │  ├─ embedding: [sparse TF-IDF]                      │
     │  ├─ metadata: {same as dense}                       │
     │  └─ document: on_screen_text + visual keywords      │
     └───────────────────┬──────────────────────────────────┘
                         │
                    (INDEXING COMPLETE)
                    (Ready for Queries)
                         │
     ┌───────────────────▼──────────────────────────────────┐
     │                                                      │
     │           PHASE 4: QUERY & ANSWER (Runtime)         │
     │           ────────────────────────────────           │
     │                                                      │
     │    USER QUERY: "What happens at 2 minutes?"        │
     │                    │                                │
     │    1. INTENT CLASSIFICATION                         │
     │       LLM analyzes query → TIMESTAMP intent         │
     │       Extract temporal hints: "2 minutes" = 120s    │
     │                    │                                │
     │    2. QUERY REWRITING (optional)                    │
     │       Semantic expansion, entity extraction         │
     │                    │                                │
     │    3. ADAPTIVE RETRIEVAL                            │
     │       ├─ Dense query embedding of rewritten query   │
     │       ├─ Search video_moments_dense (top-10)        │
     │       ├─ Search video_moments_sparse (top-10)       │
     │       ├─ Reciprocal Rank Fusion merge               │
     │       └─ Temporal proximity re-rank (near 120s)     │
     │                    │                                │
     │    RETRIEVED: [scene_5(120s), scene_6(130s), ...]   │
     │                    │                                │
     │    4. GROUNDED ANSWER GENERATION                    │
     │       ├─ Pack retrieved scenes into context prompt  │
     │       ├─ LLM (Gemini) generates answer              │
     │       ├─ Mandatory [Scene @ HH:MM:SS] citations     │
     │       └─ Output: "At 2 minutes [Scene @ 02:00], ... │
     │                    │                                │
     │    5. HALLUCINATION DETECTION & VALIDATION          │
     │       ├─ Extract claims from generated answer       │
     │       ├─ Cross-check against retrieved context      │
     │       ├─ Compute grounding_ratio                    │
     │       └─ Flag unsupported claims                    │
     │                    │                                │
     │    6. CONFIDENCE SCORING                            │
     │       score = (retrieval_sim × 0.3)                 │
     │             + (grounding_ratio × 0.5)               │
     │             + (temporal_coherence × 0.2)            │
     │                    │                                │
     │    7. CONVERSATION MEMORY UPDATE                    │
     │       Store (query, answer, context) for next turn  │
     │                    │                                │
     └───────────────────┬──────────────────────────────────┘
                         │
     ┌───────────────────▼──────────────────────────────────┐
     │  FINAL OUTPUT:                                       │
     │  ┌──────────────────────────────────────────────────┐│
     │  │ Answer: "At 2:00 [Scene @ 02:00], the speaker  ││
     │  │           introduces the architecture..."        ││
     │  │                                                  ││
     │  │ Confidence: 0.92                                ││
     │  │ Citations:                                       ││
     │  │  • [Scene @ 02:00] — scene_5_center.jpg        ││
     │  │  • [Scene @ 02:10] — scene_6_center.jpg        ││
     │  └──────────────────────────────────────────────────┘│
     │                                                      │
     │  SESSION LOG: saved to data/logs/session_<ts>.jsonl││
     │  {                                                  │
     │   "timestamp": "2026-03-17T12:34:56",              │
     │   "query": "What happens at 2 minutes?",           │
     │   "intent": "TIMESTAMP",                           │
     │   "answer": "At 2:00 [Scene @ 02:00]...",         │
     │   "confidence": 0.92,                              │
     │   "retrieved_scenes": ["scene_5", "scene_6"]       │
     │  }                                                  │
     └──────────────────────────────────────────────────────┘
```

### Multi-Turn Conversation Flow

```
┌────────────────────────────────────────────────────┐
│         Conversation Memory (Sliding Window)        │
│                                                    │
│  Turn 1: Q: "What is this video about?"          │
│          A: "This video explains..."              │
│          Context: [scenes 1-3 retrieved]          │
│                                                    │
│  Turn 2: Q: "And what about the next part?"     │
│          Context: Previous conversation + query   │
│          A: "Following that, the video shows..." │
│          Context: [scenes 3-5 retrieved]          │
│                                                    │
│  Turn n: Q: "Can you elaborate on that?"        │
│          Context: Last N turns + query            │
│          A: "In more detail, the scenes..."       │
│          Context: [dynamic retrieval with memory] │
└────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites
- Python 3.9+
- FFmpeg (for audio extraction)
- CUDA-capable GPU (optional but recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Harish-0412/RAG-Enhanced-Video-Scene-Understanding.git
cd VideoSceneRAG
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirement.txt
```

### Step 4: Configure API Keys
Create a `.env` file in the repository root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

To get a Gemini API key: https://ai.google.dev/tutorials/setup

---

## 💻 Usage

### Phase 1: Extract Frames & Scene Metadata
```bash
cd src
python phase1_sampling.py
```
**Outputs:**
- `data/processed/frames/` — extracted center frames
- `data/processed/metadata/mapping.json` — scene metadata

### Phase 2: Transcribe Audio & Extract Visual Features
```bash
python phase2_audio.py
python phase2_visual.py
```
**Outputs:**
- `data/processed/metadata/transcript.json` — ASR transcripts
- `data/processed/metadata/enriched_metadata.json` — multimodal features

### Phase 3: Build Vector Index
```bash
python phase3_indexing.py
```
**Outputs:**
- `chroma_db/` — ChromaDB vector store (persistent)
- Console output showing collection statistics

### Phase 4: Query the Video (Interactive REPL)
```bash
python main.py
```

**Interactive Shell Commands:**
```
> What is the main topic of this video?
> Tell me what happens at 3 minutes 45 seconds
> Compare scene A with scene B
> Summarize the entire video
> /history          # Show previous queries
> /stats            # Display session statistics
> /help             # Show available commands
> /quit             # Exit the program
```

### Batch Mode: Run Full Pipeline
```bash
python test_integration.py
```

---

## 📁 Project Structure

```
VideoSceneRAG/
├── README.md                          # This file
├── requirement.txt                    # Python dependencies
├── .env                               # API keys (add locally)
│
├── src/
│   ├── main.py                        # Interactive CLI interface (Phase 4)
│   ├── phase1_sampling.py             # Scene detection & frame extraction
│   ├── phase2_audio.py                # Speech-to-text (Whisper)
│   ├── phase2_visual.py               # Visual feature extraction
│   ├── phase3_indexing.py             # Vector indexing (ChromaDB)
│   ├── phase4_rag.py                  # RAG answer engine (Gemini)
│   ├── utils.py                       # Shared utilities & helpers
│   ├── mock_phase2_visual.py          # Mock visual features (for testing)
│   ├── test_integration.py            # Full pipeline integration tests
│   ├── test_phase3_search.py          # Retrieval testing
│   └── __pycache__/                   # Compiled Python files
│
├── data/
│   ├── input_videos/                  # Place your video(s) here
│   │   └── demo_video.mp4
│   │
│   ├── audio/                         # Extracted audio tracks
│   │
│   ├── processed/
│   │   ├── frames/                    # Extracted center frames (Phase 1)
│   │   │   ├── scene_001_center.jpg
│   │   │   ├── scene_002_center.jpg
│   │   │   └── ...
│   │   │
│   │   ├── audio/                     # Processed audio files
│   │   │
│   │   └── metadata/                  # Intermediate & output metadata
│   │       ├── mapping.json           # Phase 1: Scene boundaries & frames
│   │       ├── transcript.json        # Phase 2A: Raw ASR output
│   │       └── enriched_metadata.json # Phase 2B: Multimodal features
│   │
│   ├── results/                       # Final output
│   │   └── multimodal_data.json       # Consolidated results
│   │
│   └── logs/                          # Session logs
│       └── session_<timestamp>.jsonl  # Query/answer history
│
└── chroma_db/                         # Vector database (persistent)
    └── chroma.sqlite3                 # Hybrid dense + sparse index
```

---

## ⚙️ Configuration

### Key Environment Variables
| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | ✅ Yes | Google Gemini API key for answer generation |

### Tunable Hyperparameters

#### Phase 1 (scene1_sampling.py)
```python
SCENE_THRESHOLD = 25.0        # PySceneDetect content threshold
DOWNSAMPLE_FACTOR = 2         # Reduce resolution for speed
VISUAL_HASH_THRESHOLD = 0.95  # pHash similarity (deduplication)
```

#### Phase 3 (phase3_indexing.py)
```python
DEFAULT_TOP_K = 5             # Number of scenes to retrieve per collection
BATCH_SIZE = 32               # Scenes per batch during indexing
TEMPORAL_BOOST = 1.2          # Weight for temporal proximity
```

#### Phase 4 (phase4_rag.py)
```python
CONFIDENCE_WEIGHTS = {
  'retrieval': 0.3,           # Weight for retrieval similarity
  'grounding': 0.5,           # Weight for hallucination check
  'temporal': 0.2             # Weight for temporal coherence
}
CONVERSATION_MEMORY_DEPTH = 5 # Number of turns to remember
```

---

## 📊 Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| **Supported Video Length** | Up to 60 min | Size depends on available GPU/RAM |
| **Scene Detection** | ~1-2s per minute | Depends on content complexity |
| **ASR (Whisper)** | ~1-3s per minute | GPU: 5-10min/hr; CPU: 30min/hr |
| **Embedding** | ~0.1s per scene | Batch processing with SentenceTransformer |
| **Query Latency** | 1-3 seconds | Dense + sparse retrieval + LLM generation |
| **Vector DB Size** | ~1MB per 100 scenes | ChromaDB with metadata |

---

## 🔍 Troubleshooting

### Issue: "GEMINI_API_KEY not found"
**Solution:** Create `.env` file with your API key
```env
GEMINI_API_KEY=your_key_here
```

### Issue: "No module named 'faster_whisper'"
**Solution:** Reinstall dependencies
```bash
pip install --upgrade -r requirement.txt
```

### Issue: ChromaDB connection error
**Solution:** Clear and rebuild the database
```bash
rm -rf chroma_db/
python src/phase3_indexing.py
```

### Issue: Out of memory during Phase 2
**Solution:** Reduce video length or batch size
```python
# In phase2_audio.py
BATCH_SIZE = 16  # Reduce from default 32
```

---

## 🛠️ Development

### Running Tests
```bash
cd src
python test_integration.py      # Full pipeline test
python test_phase3_search.py    # Retrieval system test
```

### Extending with Custom Models
Each phase can be customized:
- **Phase 1**: Swap PySceneDetect with other detectors
- **Phase 2**: Use different ASR (Wav2Vec, Deepgram) or vision models
- **Phase 3**: Extend embeddings (multi-modal transformers)
- **Phase 4**: Use different LLMs (GPT-4, Claude)

---

## 📚 Technical Highlights

### Advanced Techniques Implemented

1. **Perceptual Hashing (pHash)**: Near-duplicate frame detection
2. **Reciprocal Rank Fusion (RRF)**: Hybrid search result merging
3. **Temporal Re-ranking**: Context-aware result ordering
4. **Entity Extraction**: NLP-driven query understanding
5. **Grounding Checks**: Hallucination detection & mitigation
6. **Confidence Scoring**: Composite quality metrics
7. **Conversation Memory**: Multi-turn context preservation

### Research Papers & References
- **Scene Detection**: PySceneDetect algorithm (ContentDetector)
- **Speech Recognition**: Faster Whisper optimization paper
- **Embeddings**: Sentence-BERT (Sentence Transformers)
- **Vector Search**: Reciprocal Rank Fusion algorithm
- **RAG**: "Retrieval-Augmented Generation for Large Language Models" (Lewis et al., 2020)

---

## 📝 License

This project is licensed under the MIT License. See LICENSE file for details.

---

## 👨‍💻 Authors & Contributors

- **Harish-0412** — Project Creator, Research & Lead Developer
- **Dharsan6** — UI and UX
- **AkashB-Glitch** - Backend and API

---

## 🙋 Support

For issues, feature requests, or questions:
1. Open an Issue on GitHub: [GitHub Issues](https://github.com/Harish-0412/RAG-Enhanced-Video-Scene-Understanding/issues)
2. Review the troubleshooting section above
3. Check existing documentation in the code comments

---

## 🎓 Educational Use

This project is ideal for learning:
- **Computer Vision**: Scene detection, frame extraction
- **NLP**: Query understanding, answer generation
- **Information Retrieval**: Vector search, hybrid ranking
- **System Design**: Multi-phase ML pipelines
- **LLM Integration**: RAG patterns, prompt engineering

---

## 🔮 Future Enhancements

Planned features for future releases:
- [ ] Streaming video support
- [ ] Multi-language support
- [ ] Real-time indexing (process video while playing)
- [ ] Web UI dashboard
- [ ] Batch query processing
- [ ] Custom fine-tuned models
- [ ] Advanced filtering (speaker identification, sentiment)
- [ ] Video summarization
- [ ] Interactive clip generation

---

**Last Updated:** March 17, 2026
