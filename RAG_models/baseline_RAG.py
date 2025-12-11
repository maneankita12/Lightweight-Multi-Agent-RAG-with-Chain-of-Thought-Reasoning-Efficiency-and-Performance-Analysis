import pickle
import faiss
from sentence_transformers import SentenceTransformer
import requests
import re

BASE_PATH = '/Users/ankitamane/Documents/NEU/SEM1/NLP'
DB_PATH = f'{BASE_PATH}/final_rag_database'

OLLAMA_URL = 'http://localhost:11434'
MODEL_NAME = 'llama3.2'

#load documents
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index(f'{DB_PATH}/faiss_index.bin')

with open(f'{DB_PATH}/documents.pkl', 'rb') as f:
    documents = pickle.load(f)

def retrieve(query, k=3):
    emb = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    scores, indices = index.search(emb, k=k)
    
    contexts = []
    for idx, score in zip(indices[0], scores[0]):
        if score > 0.3:
            contexts.append(documents[int(idx)]['text'])
    
    return "\n\n".join(contexts) if contexts else None

def generate(prompt):
    try:
        resp = requests.post(
            f'{OLLAMA_URL}/api/generate',
            json={'model': MODEL_NAME, 'prompt': prompt, 'stream': False,
                  'options': {'temperature': 0.1, 'num_predict': 50}},
            timeout=30
        )
        if resp.status_code == 200:
            text = resp.json()['response'].strip()
            return text.split('.')[0].strip()
        return None
    except:
        return None

def baseline_rag(query, show_context=False):
    context = retrieve(query, k=3)
    
    if not context:
        return "No relevant information found."
    
    if show_context:
        print(f"\n📄 Retrieved Context:\n{context[:300]}...\n")
    
    prompt = f"""Answer the question based on the context. Give only the direct answer, no explanation.

Context: {context}

Question: {query}

Answer:"""
    
    answer = generate(prompt)
    
    return answer if answer else "Failed to generate answer."


print("  - Type your question")
print("  - Type 'context' to see retrieved context")
print("  - Type 'exit' to quit")

show_context = False

while True:
    query = input(" Your question: ").strip()
    
    if query.lower() in ['exit', 'quit', 'q']:
        print("\nGoodbye!")
        break
    
    if query.lower() == 'context':
        show_context = not show_context
        print(f"Context display: {'ON' if show_context else 'OFF'}\n")
        continue
    
    if not query:
        continue
    
    print("\nAnswer: ", end="", flush=True)
    answer = baseline_rag(query, show_context=show_context)
    print(answer + "\n")