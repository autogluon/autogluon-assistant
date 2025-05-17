"""
Climate Fever Retrieval System

This script implements a retrieval system for the Climate Fever dataset using FlagEmbedding.
It retrieves the top 10 most relevant documents for each query in the test set.

Additional installation requirements:
- pip install FlagEmbedding
- pip install faiss-cpu (or faiss-gpu for GPU support)
- pip install pandas numpy tqdm
"""

import json
import os

import faiss
import numpy as np
import pandas as pd
from FlagEmbedding import FlagModel
from tqdm import tqdm

OUTPUT_DIR = "./"

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load corpus data
    print("Loading corpus data...")
    corpus = []
    corpus_ids = []
    with open('/media/agent/maab/datasets/climate_fever/training/corpus.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            # Combine title and text for better retrieval
            corpus.append(f"{data['title']}: {data['text']}")
            corpus_ids.append(data['_id'])
    
    # Load queries data
    print("Loading queries data...")
    queries = {}
    with open('/media/agent/maab/datasets/climate_fever/training/queries.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            queries[int(data['_id'])] = data['text']
    
    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv('/media/agent/maab/datasets/climate_fever/training/test.tsv', sep='\t')
    test_queries = test_df['query-id'].tolist()
    
    # Initialize the embedding model
    print("Initializing embedding model...")
    model = FlagModel(
        'BAAI/bge-base-en-v1.5',
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
        use_fp16=True
    )
    
    # Generate corpus embeddings
    print("Generating corpus embeddings...")
    # Process in batches to avoid memory issues
    batch_size = 1024
    corpus_embeddings = []
    
    for i in tqdm(range(0, len(corpus), batch_size)):
        batch = corpus[i:i+batch_size]
        emb = model.encode(batch, convert_to_numpy=True)
        corpus_embeddings.append(emb)
    
    corpus_embeddings = np.vstack(corpus_embeddings)
    corpus_embeddings = corpus_embeddings.astype(np.float32)  # Faiss requires float32
    
    # Create Faiss index
    print("Creating Faiss index...")
    dim = corpus_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity with normalized vectors
    index.add(corpus_embeddings)
    
    # Generate query embeddings and search
    print("Generating query embeddings and searching...")
    results = []
    
    for query_id in tqdm(test_queries):
        query_text = queries[query_id]
        query_embedding = model.encode_queries([query_text], convert_to_numpy=True)
        
        # Search top 10 documents
        scores, indices = index.search(query_embedding, k=10)
        
        # Add results to the list
        for i, (doc_idx, score) in enumerate(zip(indices[0], scores[0])):
            results.append({
                'query-id': query_id,
                'corpus-id': corpus_ids[doc_idx],
                'score': float(score)
            })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to TSV file
    output_path = os.path.join(OUTPUT_DIR, "results.tsv")
    results_df.to_csv(output_path, sep='\t', index=False)
    
    print(f"Results saved to {output_path}")