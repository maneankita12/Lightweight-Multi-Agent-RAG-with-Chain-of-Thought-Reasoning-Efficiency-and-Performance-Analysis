# Lightweight Multi-Agent RAG with Chain-of-Thought Reasoning

##  Project Description

This project implements and evaluates a comprehensive Retrieval-Augmented Generation (RAG) system enhanced with Chain-of-Thought (CoT) reasoning and Multi-Agent (MA) architectures. We systematically compare three progressive architectures (Baseline RAG, CoT-RAG, and MA-RAG) across multiple large language models including Ollama llama3.2 (3B), Groq Llama models (70B), and the Gemini family (2.0-flash, 2.5-flash variants) on three benchmark datasets: HotpotQA, Natural Questions, and FEVER.

##  Key Features

- **Progressive Architecture Comparison**: Baseline RAG → CoT-RAG → MA-RAG
- **Multi-Model Evaluation**: Tested across 3 different LLMs ranging multiple parameters
- **Comprehensive Benchmarking**: Evaluated on multi-hop reasoning, factual QA, and fact verification tasks
- **Scalable Vector Database**: 795,768 Wikipedia documents with FAISS indexing
- **Efficient Multi-Agent System**: Achieves 40% latency reduction while maintaining 84-86% F1 scores

##  Related Work

Our Multi-Agent RAG implementation builds upon the foundational work presented in:
- **"MA-RAG: Multi-Agent Retrieval-Augmented Generation via Collaborative Chain-of-Thought Reasoning"** (Liu et al., 2024)
- **"CoT-RAG: Integrating Chain of Thought and Retrieval-Augmented Generation to Enhance Reasoning in Large Language Models"** (Zhang et al., 2024)

##  Team Members

- **Ankita Anil Mane** (NUID: 002540892) 
- **Khyati Nirenkumar Amin** (NUID: 002511574)
- **Raj Gupta** (NUID: 002068701)

##  Acknowledgments

We thank the Northeastern University CS6120 Natural Language Processing Course Staff for their guidance, and Anthropic, Google, and Groq for providing API access for model evaluation. Special thanks to the open-source NLP community for the foundational tools and datasets that made this research possible.

---
