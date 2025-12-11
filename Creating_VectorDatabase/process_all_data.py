import json
import csv
import os
from tqdm import tqdm

BASE_PATH = '/Users/ankitamane/Documents/NEU/SEM1/NLP'
RAW_DATA_PATH = f'{BASE_PATH}/dataset'
OUTPUT_PATH = f'{BASE_PATH}/processed_data'

os.makedirs(OUTPUT_PATH, exist_ok=True)

all_documents = []
seen_texts = set()

# 1. process Hotpot QA

print("\n1. Processing HotpotQA files...")

hotpot_files = [
    'hotpot_train_v1.1.json',
    'hotpot_dev_distractor_v1.json',
    'hotpot_dev_fullwiki_v1.json',
    'hotpot_test_fullwiki_v1.json'
]

hotpot_count = 0

for filename in hotpot_files:
    filepath = os.path.join(RAW_DATA_PATH, filename)
    
    if not os.path.exists(filepath):
        print(f" {filename} not found, skipping...")
        continue
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Debug: Check first example structure
    if len(data) > 0:
        first_context = data[0].get('context', None)
        print(f"  DEBUG - Context type: {type(first_context)}")
        
        if isinstance(first_context, list) and len(first_context) > 0:
            print(f"  DEBUG - First context item type: {type(first_context[0])}")
            print(f"  DEBUG - First context item: {first_context[0]}")
        elif isinstance(first_context, dict):
            print(f"  DEBUG - Context keys: {first_context.keys()}")
    
    for idx, item in enumerate(tqdm(data, desc=f"  {filename}")):
        context = item.get('context', [])
        
        # Handle list of [title, sentences] pairs
        if isinstance(context, list):
            for ctx_item in context:
                if isinstance(ctx_item, list) and len(ctx_item) >= 2:
                    title = ctx_item[0]
                    sentences = ctx_item[1]
                    
                    # Sentences should be a list
                    if isinstance(sentences, list):
                        text = f"{title}: {' '.join(sentences)}"
                    else:
                        text = f"{title}: {sentences}"
                    
                    if text not in seen_texts and len(text) > 50:
                        seen_texts.add(text)
                        all_documents.append({
                            'id': f"hotpot_{hotpot_count}",
                            'text': text,
                            'source': 'hotpotqa',
                            'title': title,
                            'type': 'context_paragraph'
                        })
                        hotpot_count += 1
        
        # Handle dict structure 
        elif isinstance(context, dict):
            titles = context.get('title', [])
            sentences_list = context.get('sentences', [])
            
            for title, sentences in zip(titles, sentences_list):
                if isinstance(sentences, list):
                    text = f"{title}: {' '.join(sentences)}"
                else:
                    text = f"{title}: {sentences}"
                
                if text not in seen_texts and len(text) > 50:
                    seen_texts.add(text)
                    all_documents.append({
                        'id': f"hotpot_{hotpot_count}",
                        'text': text,
                        'source': 'hotpotqa',
                        'title': title,
                        'type': 'context_paragraph'
                    })
                    hotpot_count += 1

print(f"\n✓ HotpotQA total: {hotpot_count} paragraphs")


# 2. Process Natural Questions

print("\n2. Processing Natural Questions...")

nq_files = [
    'Natural-Questions-Base.csv',
    'nq_small.csv'
]

nq_count = 0

for filename in nq_files:
    filepath = os.path.join(RAW_DATA_PATH, filename)
    
    if not os.path.exists(filepath):
        print(f" {filename} not found, skipping...")
        continue
        
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline()
            f.seek(0)
            
            # Use csv.Sniffer to detect format
            try:
                dialect = csv.Sniffer().sniff(first_line)
                reader = csv.DictReader(f, dialect=dialect)
            except:
                reader = csv.DictReader(f)
            
            for row in tqdm(reader, desc=f"  {filename}"):
                # Try different possible column names
                context = (
                    row.get('document_text', '') or 
                    row.get('context', '') or
                    row.get('passage', '') or
                    row.get('answer', '')
                )
                
                if context and len(context) > 50:
                    text = context[:2000]  # Limit length
                    
                    if text not in seen_texts:
                        seen_texts.add(text)
                        all_documents.append({
                            'id': f"nq_{nq_count}",
                            'text': text,
                            'source': 'natural_questions',
                            'type': 'passage'
                        })
                        nq_count += 1
    
    except Exception as e:
        print(f"  Error processing {filename}: {e}")

print(f"Natural Questions total: {nq_count} passages")


#3. Process Fever Dataset

print("\n3. Processing FEVER...")

fever_filepath = os.path.join(RAW_DATA_PATH, 'FEVER.json')
fever_count = 0

if os.path.exists(fever_filepath):
    
    with open(fever_filepath, 'r', encoding='utf-8') as f:
        fever_data = json.load(f)
    
    for item in tqdm(fever_data, desc="  FEVER"):
        # Extract all evidence text
        evidence_list = item.get('evidence', [])
        
        if isinstance(evidence_list, list):
            for evidence_set in evidence_list:
                if isinstance(evidence_set, list):
                    for evidence in evidence_set:
                        # Evidence format: [wiki_url, sent_id, text, ...]
                        if isinstance(evidence, (list, tuple)) and len(evidence) >= 3:
                            evidence_text = evidence[2]
                            
                            if evidence_text and len(evidence_text) > 20:
                                text = f"Evidence: {evidence_text}"
                                
                                if text not in seen_texts:
                                    seen_texts.add(text)
                                    all_documents.append({
                                        'id': f"fever_{fever_count}",
                                        'text': text,
                                        'source': 'fever',
                                        'type': 'evidence'
                                    })
                                    fever_count += 1

print(f"FEVER total: {fever_count} evidence passages")

# Process Wikipedia

print("\n4. Processing Wikipedia DPR passages...")

wiki_filepath = os.path.join(RAW_DATA_PATH, 'psgs_w100.tsv')
wiki_count = 0
WIKI_TARGET = 500000 - (hotpot_count + nq_count + fever_count)  # Fill to ~500K

if WIKI_TARGET < 0:
    WIKI_TARGET = 50000  # Minimum 50K

print(f"  Target Wikipedia passages: {WIKI_TARGET}")

if os.path.exists(wiki_filepath):
    
    with open(wiki_filepath, 'r', encoding='utf-8') as f:
        # Skip header
        header = next(f)
        
        for line in tqdm(f, total=WIKI_TARGET, desc="  Wikipedia"):
            if wiki_count >= WIKI_TARGET:
                break
            
            try:
                parts = line.strip().split('\t')
                
                if len(parts) >= 2:
                    passage_text = parts[1]
                    title = parts[2] if len(parts) > 2 else "Wikipedia"
                    
                    if len(passage_text) > 50:
                        text = f"{title}: {passage_text}"
                        
                        if text not in seen_texts:
                            seen_texts.add(text)
                            all_documents.append({
                                'id': f"wiki_{wiki_count}",
                                'text': text,
                                'source': 'wikipedia_dpr',
                                'title': title,
                                'type': 'wikipedia_passage'
                            })
                            wiki_count += 1
            
            except Exception as e:
                continue

print(f" Wikipedia total: {wiki_count} passages")


print(f"\n Total Documents : {len(all_documents)}")
print(f"\n Breakdown by source:")
print(f"  HotpotQA: {hotpot_count:,}")
print(f"  Natural Questions: {nq_count:,}")
print(f"  FEVER: {fever_count:,}")
print(f"  Wikipedia: {wiki_count:,}")

# Save
output_file = f'{OUTPUT_PATH}/all_documents.json'

print(f"\nSaving to {output_file}...")

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_documents, f)

print(f" Saved {len(all_documents):,} documents")
   