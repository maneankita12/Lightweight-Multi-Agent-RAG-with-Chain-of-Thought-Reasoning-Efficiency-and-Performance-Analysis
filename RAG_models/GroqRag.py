import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from groq import Groq
import os
import re
import pickle
from typing import List, Dict, Optional
from tqdm import tqdm
import json

# Evaluation metrics
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

nltk.download('punkt', quiet=True)


# Initialize Groq client
class GroqRAGSystem:
    def __init__(self, groq_api_key: str, faiss_index_path: str = None, 
                 embedding_model: str = "all-MiniLM-L6-v2"):
        
        self.client = Groq(api_key=groq_api_key)
        
        print("Loading embedding model")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Load your existing FAISS index
        if faiss_index_path and os.path.exists(faiss_index_path):
            print(f"Loading FAISS index from {faiss_index_path}...")
            self.index = faiss.read_index(faiss_index_path)
            print(f"Index loaded with {self.index.ntotal} vectors")
        else:
            print("No FAISS index provided. You'll need to create one.")
            self.index = None
        
        self.documents = []
        
    def load_documents(self, documents: List[str]):
        self.documents = documents
    
    def load_documents_from_pickle(self, pickle_path: str):
        with open(pickle_path, 'rb') as f:
            self.documents = pickle.load(f)
    
    def create_index(self, documents: List[str]):
        self.documents = documents
        embeddings = self.embedding_model.encode(
            documents, 
            show_progress_bar=True,
            batch_size=32
        )
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
    def save_index(self, path: str):
        if self.index:
            faiss.write_index(self.index, path)
    
    def retrieve(self, query: str, k: int = 3) -> str:
        if self.index is None:
            raise ValueError("No FAISS index loaded!")
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        
        # Search
        distances, indices = self.index.search(
            query_embedding.astype('float32'), k
        )
        
        # Get documents
        retrieved_docs = []
        for idx in indices[0]:
            doc = self.documents[idx]
            # Handle if documents are dictionaries
            if isinstance(doc, dict):
                # Try common keys for document text
                if 'text' in doc:
                    retrieved_docs.append(doc['text'])
                elif 'content' in doc:
                    retrieved_docs.append(doc['content'])
                elif 'context' in doc:
                    retrieved_docs.append(doc['context'])
                elif 'document' in doc:
                    retrieved_docs.append(doc['document'])
                else:
                    # If none of these keys, convert entire dict to string
                    retrieved_docs.append(str(doc))
            else:
                # Documents are already strings
                retrieved_docs.append(str(doc))
        
        # Combine into context
        context = "\n\n".join(retrieved_docs)
        return context
    
   # Generate response using Groq
    def generate_groq(self, prompt: str, model: str = "llama-3.1-8b-instant",
                     temperature: float = 0.1, max_tokens: int = 100) -> str:
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error with Groq: {e}")
            return ""
    
    def extract_answer(self, text: str) -> str:
        if not text:
            return ""
        
        # Remove common prefixes
        text = re.sub(r'^(Answer:|A:)\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^(The answer is|It is|This is)\s+', '', text, flags=re.IGNORECASE)
        
        # Get first sentence
        sentences = re.split(r'[.!?]', text)
        answer = sentences[0].strip() if sentences else text
        
        # If still too long, extract key phrase
        if len(answer.split()) > 15:
            match = re.search(r'(?:is|are|was|were)\s+([^,]+?)(?:\.|,|$)', answer)
            if match:
                return match.group(1).strip()
        
        return answer
    

    # Baseline RAG
    def baseline_rag(self, query: str) -> Optional[str]:
        context = self.retrieve(query, k=3)  # Increase to k=3 for fairer comparison
        
        if not context:
            return None
        
        prompt = f"""Answer the question based on the context. Give only the direct answer, no explanation.

Context: {context}

Question: {query}

Answer:"""
        
        response = self.generate_groq(prompt, temperature=0.1, max_tokens=50)
        
        if not response:
            return ""
        
        # Extract first sentence like your baseline
        answer = response.split('.')[0].strip()
        return self.extract_answer(answer)
    
    
    # RAG + Minimal COT
    def rag_zero_shot_cot(self, query: str) -> Optional[str]:
        """RAG with minimal modification - just add 'carefully'"""
        context = self.retrieve(query, k=3)
        
        if not context:
            return None
        
        prompt = f"""Answer the question based on the context. Read the context carefully and give only the direct answer, no explanation.

Context: {context}

Question: {query}

Answer:"""
        
        response = self.generate_groq(prompt, temperature=0.1, max_tokens=50)
        
        if not response:
            return ""
        
        answer = response.split('.')[0].strip()
        return self.extract_answer(answer)
    
   
    # RAG + few shot COT
    def rag_few_shot_cot(self, query: str) -> Optional[str]:
        """RAG with minimal few-shot examples"""
        context = self.retrieve(query, k=3)
        
        if not context:
            return None
        
        prompt = f"""Answer the question based on the context. Give only the direct answer, no explanation.

Example:
Context: Paris is the capital of France.
Question: What is the capital of France?
Answer: Paris

Example:
Context: Python was created by Guido van Rossum.
Question: Who created Python?
Answer: Guido van Rossum

Context: {context}

Question: {query}

Answer:"""
        
        response = self.generate_groq(prompt, temperature=0.1, max_tokens=50)
        
        if not response:
            return ""
        
        answer = response.split('.')[0].strip()
        return self.extract_answer(answer)



# Evaluatiob Metrics
def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def compute_exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)
    
    common = set(pred_tokens) & set(gt_tokens)
    
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1



# Dataset Evaluations
def evaluate_hotpotqa(rag_system: GroqRAGSystem, num_samples: int = 100):
    print("EVALUATING ON HOTPOTQA")
    
    dataset = load_dataset('hotpot_qa', 'distractor', split='validation')
    dataset = dataset.shuffle(seed=42).select(range(num_samples))
    
    methods = {
        'Baseline RAG': rag_system.baseline_rag,
        'RAG + Zero-Shot CoT': rag_system.rag_zero_shot_cot,
        'RAG + Few-Shot CoT': rag_system.rag_few_shot_cot
    }
    
    results = {}
    
    for method_name, method_func in methods.items():
        print(f"\nEvaluating {method_name}...")
        em_scores = []
        f1_scores = []
        accuracy_scores = []
        
        for example in tqdm(dataset, desc=method_name):
            prediction = method_func(example['question'])
            
            if prediction is None:
                prediction = ""
            
            em = compute_exact_match(prediction, example['answer'])
            f1 = compute_f1(prediction, example['answer'])
            
            em_scores.append(em)
            f1_scores.append(f1)
            # For QA tasks, accuracy = exact match
            accuracy_scores.append(em)
        
        results[method_name] = {
            'exact_match': np.mean(em_scores) * 100,
            'f1_score': np.mean(f1_scores) * 100,
            'accuracy': np.mean(accuracy_scores) * 100,
            'num_samples': len(em_scores)
        }
    
    return results


def evaluate_nq_open(rag_system: GroqRAGSystem, num_samples: int = 100):
    print("EVALUATING ON NATURAL QUESTIONS")
    
    dataset = load_dataset('nq_open', split='validation')
    dataset = dataset.shuffle(seed=42).select(range(num_samples))
    
    methods = {
        'Baseline RAG': rag_system.baseline_rag,
        'RAG + Zero-Shot CoT': rag_system.rag_zero_shot_cot,
        'RAG + Few-Shot CoT': rag_system.rag_few_shot_cot
    }
    
    results = {}
    
    for method_name, method_func in methods.items():
        print(f"\nEvaluating {method_name}...")
        em_scores = []
        f1_scores = []
        accuracy_scores = []
        
        for example in tqdm(dataset, desc=method_name):
            prediction = method_func(example['question'])
            
            if prediction is None:
                prediction = ""
            
            # NQ has multiple possible answers
            em_all = [compute_exact_match(prediction, ans) for ans in example['answer']]
            f1_all = [compute_f1(prediction, ans) for ans in example['answer']]
            
            em = max(em_all)
            f1 = max(f1_all)
            
            em_scores.append(em)
            f1_scores.append(f1)
            # For QA tasks, accuracy = exact match
            accuracy_scores.append(em)
        
        results[method_name] = {
            'exact_match': np.mean(em_scores) * 100,
            'f1_score': np.mean(f1_scores) * 100,
            'accuracy': np.mean(accuracy_scores) * 100,
            'num_samples': len(em_scores)
        }
    
    return results


def evaluate_fever(rag_system: GroqRAGSystem, num_samples: int = 100):
    print("EVALUATING ON FEVER")
    
    dataset = load_dataset('fever', 'v1.0', split='labelled_dev')
    # Filter only SUPPORTS and REFUTES claims (skip NOT ENOUGH INFO)
    dataset = dataset.filter(lambda x: x['label'] in ['SUPPORTS', 'REFUTES'])
    dataset = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))
    
    methods = {
        'Baseline RAG': rag_system.baseline_rag,
        'RAG + Zero-Shot CoT': rag_system.rag_zero_shot_cot,
        'RAG + Few-Shot CoT': rag_system.rag_few_shot_cot
    }
    
    results = {}
    
    for method_name, method_func in methods.items():
        print(f"\nEvaluating {method_name}...")
        em_scores = []
        f1_scores = []
        accuracy_scores = []
        
        for example in tqdm(dataset, desc=method_name):
            # Create question from claim
            claim = example['claim']
            true_label = example['label']
            
            # Ask model to verify the claim
            question = f"Is this claim true or false: {claim}"
            prediction = method_func(question)
            
            if prediction is None:
                prediction = ""
            
            # Ground truth answer based on label
            if true_label == 'SUPPORTS':
                ground_truth = "true"
            else:  # REFUTES
                ground_truth = "false"
            
            # Compute EM and F1
            em = compute_exact_match(prediction, ground_truth)
            f1 = compute_f1(prediction, ground_truth)
            
            em_scores.append(em)
            f1_scores.append(f1)
            
            # Compute accuracy (classification correctness)
            pred_lower = prediction.lower()
            if true_label == 'SUPPORTS':
                is_correct = any(word in pred_lower for word in ['true', 'correct', 'yes', 'support'])
            else:  # REFUTES
                is_correct = any(word in pred_lower for word in ['false', 'incorrect', 'no', 'refute'])
            
            accuracy_scores.append(float(is_correct))
        
        results[method_name] = {
            'exact_match': np.mean(em_scores) * 100,
            'f1_score': np.mean(f1_scores) * 100,
            'accuracy': np.mean(accuracy_scores) * 100,
            'num_samples': len(em_scores)
        }
    
    return results


def print_results(results: Dict, dataset_name: str):
    print(f"RESULTS FOR {dataset_name.upper()}")
    
    for method_name, scores in results.items():
        print(f"{method_name}:")
        print(f"  Exact Match: {scores['exact_match']:.2f}%")
        print(f"  F1 Score:    {scores['f1_score']:.2f}%")
        print(f"  Accuracy:    {scores['accuracy']:.2f}%")
        print(f"  Samples:     {scores['num_samples']}")
        print()



# Main Execution
if __name__ == "__main__":
    
    GROQ_API_KEY = "groq_api" #removed while committing on git  
    FAISS_INDEX_PATH = "/Users/ankitamane/Documents/NEU/SEM1/NLP/final_rag_database/faiss_index.bin"  
    DOCUMENTS_PICKLE_PATH = "/Users/ankitamane/Documents/NEU/SEM1/NLP/final_rag_database/documents.pkl"  
    NUM_SAMPLES = 100 
    
    
    print("Initializing Groq RAG System...")
    rag_system = GroqRAGSystem(
        groq_api_key=GROQ_API_KEY,
        faiss_index_path=FAISS_INDEX_PATH
    )
    
    rag_system.load_documents_from_pickle(DOCUMENTS_PICKLE_PATH)
    
    print("Evaluating on all datasets:")
    print(f"Samples per dataset: {NUM_SAMPLES}")
    
    all_results = {}
    
    # 1. Evaluate on HotpotQA
    try:
        hotpotqa_results = evaluate_hotpotqa(rag_system, num_samples=NUM_SAMPLES)
        print_results(hotpotqa_results, "HotpotQA")
        all_results['HotpotQA'] = hotpotqa_results
    except Exception as e:
        print(f"Error evaluating HotpotQA: {e}")
    
    # 2. Evaluate on Natural Questions
    try:
        nq_results = evaluate_nq_open(rag_system, num_samples=NUM_SAMPLES)
        print_results(nq_results, "Natural Questions")
        all_results['Natural_Questions'] = nq_results
    except Exception as e:
        print(f"Error evaluating Natural Questions: {e}")
    
    # 3. Evaluate on FEVER
    try:
        fever_results = evaluate_fever(rag_system, num_samples=NUM_SAMPLES)
        print_results(fever_results, "FEVER")
        all_results['FEVER'] = fever_results
    except Exception as e:
        print(f"Error evaluating FEVER: {e}")
    
 
    #Save results
    with open('groq_rag_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Results saved to 'groq_rag_results.json'")
    
  
    print("Comparision table: ")
    
    for dataset_name, dataset_results in all_results.items():
        print(f"\n{dataset_name}:")
        for method_name, scores in dataset_results.items():
            print(f"  {method_name:25s} | EM: {scores['exact_match']:.2f}% | F1: {scores['f1_score']:.2f}% | Acc: {scores['accuracy']:.2f}%")
    