import sys
with open('corpus/embed_corpus.py', 'r') as f:
    content = f.read()
    
# Add limit after doc_c line
if 'doc_c = min(doc_c,' not in content:
    content = content.replace(
        'doc_c = ir_dataset.docs_count()',
        'doc_c = ir_dataset.docs_count()\n    doc_c = min(doc_c, 1000000)  # Test with 1M docs'
    )
    
with open('corpus/embed_corpus.py', 'w') as f:
    f.write(content)
print("✅ Limited to 1M documents")
