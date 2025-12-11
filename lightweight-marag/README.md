# Lightweight Multi-Agent RAG with Chain-of-Thought Reasoning

**CS 6120 NLP Course Project - Fall 2025**

Team Members:
- Ankita Anil Mane (mane.anki@northeastern.edu)
- Khyati Nirenkumar Amin (amin.kh@northeastern.edu)
- Raj Gupta (gupta.r1@northeastern.edu)

---

## 🎯 Project Overview

This project implements a lightweight multi-agent Retrieval-Augmented Generation (RAG) system using small language models (< 500M parameters total) to achieve efficient question answering with explicit chain-of-thought reasoning.

**Key Features:**
- Multi-agent architecture with < 500M parameters
- Works on consumer hardware (< 3GB memory)
- Evaluated on HotpotQA, FEVER, and Natural Questions
- Achieves 75-85% of large model performance at 10% computational cost

---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/rajgupta2965/NLP.git
cd NLP/lightweight-marag
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Setup Environment Variables
```bash
# For Ollama (Free Local LLM)
echo "MODEL_NAME=llama3.1:8b" > .env

# OR for Gemini (Free Cloud API - Recommended)
echo "GOOGLE_API_KEY=your-key" > .env
echo "MODEL_NAME=gemini-1.5-flash" >> .env
```

### 5. Download Pre-computed Embeddings

**Option A: Use Shared (Recommended)**
```bash
# Download from Google Drive: [Link will be shared]
tar -xzf embeddings_250k.tar.gz
```

**Option B: Generate Your Own (4-6 hours)**
```bash
python corpus/embed_corpus.py
# This will create embeddings for 250K Wikipedia documents
```

---

## 🎮 Running Experiments

### Test Run (2 questions)
```bash
python main.py --model gemini-flash --dataset hotpotqa --exp plan_rag_extract --gpus 0 --start_index 0 --end_index 2
```

### Different Models
```bash
# Lightweight (current setup)
python main.py --model llama3-8B --dataset hotpotqa --exp plan_rag_extract --gpus 0 --start_index 0 --end_index 100

# Gemini Flash (faster)
python main.py --model gemini-flash --dataset hotpotqa --exp plan_rag_extract --gpus 0 --start_index 0 --end_index 100
```

---

## 📁 Project Structure
```
lightweight-marag/
├── agents/               # Multi-agent implementations
│   ├── plan.py          # Planning agent
│   ├── rag.py           # RAG agent
│   ├── step_definer.py  # Step definition agent
│   └── plan_executor.py # Execution coordinator
├── corpus/              # Corpus and retrieval
│   ├── embed_corpus.py  # Embedding generation
│   └── retrieve.py      # FAISS retrieval
├── src/                 # Utilities
│   ├── utils.py         # Helper functions
│   └── prompt_template.py # Prompt templates
├── emb_corpus/          # Pre-computed embeddings (not in git)
│   └── gte-ml-base/
│       └── dpr100_1953  # 250K document embeddings
├── main.py              # Main entry point
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

---

## 📊 Expected Results

Results will be saved in: `plan_rag_extract_{model}_{dataset}/`

Each JSON file contains:
```json
{
  "original_question": "Question text",
  "plan": ["sub-question 1", "sub-question 2"],
  "past_exp": [
    {
      "step": "sub-question",
      "retrieved_docs": [...],
      "answer": "answer to sub-question"
    }
  ]
}
```

---

## 📧 Contact

- Ankita: mane.anki@northeastern.edu
- Khyati: amin.kh@northeastern.edu  
- Raj: gupta.r1@northeastern.edu
