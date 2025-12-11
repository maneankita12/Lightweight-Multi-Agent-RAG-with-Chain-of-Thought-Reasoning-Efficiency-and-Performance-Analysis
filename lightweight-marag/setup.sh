#!/bin/bash

echo "🚀 Setting up Lightweight MA-RAG..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p emb_corpus/gte-ml-base
mkdir -p data
mkdir -p results
mkdir -p logs

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Get embeddings from team (emb_corpus/gte-ml-base/dpr100_1953)"
echo "2. Setup .env file with API keys"
echo "3. Run test: python main.py --model gemini-flash --dataset hotpotqa --exp plan_rag_extract --gpus 0 --start_index 0 --end_index 2"
