#!/usr/bin/env python
"""
FIQA-BEIR Retrieval Script

This script uses FlagEmbedding to train a retrieval model on the FIQA-BEIR dataset
and generate the top 10 results for each query in the test set.

Installation requirements:
pip install FlagEmbedding faiss-cpu pandas numpy tqdm
"""

import json
import os

import faiss
import numpy as np
import pandas as pd
from FlagEmbedding import FlagModel
from tqdm import tqdm

if __name__ == "__main__":
    # Define paths
    output_dir = "./"
    corpus_path = "/media/agent/maab/datasets/fiqabeir/training/corpus.jsonl"
    queries_path = "/media/agent/maab/datasets/fiqabeir/training/queries.jsonl"
    train_path = "/media/agent/maab/datasets/fiqabeir/training/train.tsv"
    test_path = "/media/agent/maab/datasets/fiqabeir/training/test.tsv"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading corpus...")
    # Load corpus
    corpus = {}
    with open(corpus_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            # Convert _id to string to ensure consistent typing
            corpus[str(item['_id'])] = item['text']
    
    print("Loading queries...")
    # Load queries
    queries = {}
    with open(queries_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            # Convert _id to string to ensure consistent typing
            queries[str(item['_id'])] = item['text']
    
    print("Loading test queries...")
    # Load test queries
    test_df = pd.read_csv(test_path, sep='\t')
    test_query_ids = [str(qid) for qid in test_df['query-id'].tolist()]
    
    # Initialize the model
    print("Initializing FlagModel...")
    model = FlagModel('BAAI/bge-base-en-v1.5', 
                      query_instruction_for_retrieval="Represent this sentence for searching relevant passages:", 
                      use_fp16=True)
    
    print("Encoding corpus...")
    # Prepare corpus texts and IDs
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[doc_id] for doc_id in corpus_ids]
    
    # Encode corpus in batches to manage memory
    batch_size = 512
    corpus_embeddings = []
    
    for i in tqdm(range(0, len(corpus_texts), batch_size)):
        batch_texts = corpus_texts[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts)
        corpus_embeddings.append(batch_embeddings)
    
    corpus_embeddings = np.vstack(corpus_embeddings)
    
    print("Building FAISS index...")
    # Build FAISS index
    dim = corpus_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity with normalized vectors
    corpus_embeddings = corpus_embeddings.astype(np.float32)  # Convert to float32 for FAISS
    index.add(corpus_embeddings)
    
    print("Encoding test queries...")
    # Prepare test queries
    test_query_texts = []
    for qid in test_query_ids:
        if qid in queries:
            test_query_texts.append(queries[qid])
        else:
            # Handle case where query ID might be an integer in one file and string in another
            int_qid = str(int(float(qid)))
            if int_qid in queries:
                test_query_texts.append(queries[int_qid])
            else:
                print(f"Warning: Query ID {qid} not found in queries data")
                test_query_texts.append("")  # Add empty string as placeholder
    
    # Encode test queries
    test_query_embeddings = model.encode_queries(test_query_texts)
    
    print("Retrieving top results...")
    # Search top 10 results for each query
    k = 10
    results = []
    
    for i, qid in enumerate(test_query_ids):
        query_embedding = test_query_embeddings[i].reshape(1, -1).astype(np.float32)
        scores, indices = index.search(query_embedding, k)
        
        for j in range(k):
            corpus_idx = indices[0][j]
            score = scores[0][j]
            corpus_id = corpus_ids[corpus_idx]
            results.append({
                'query-id': qid,
                'corpus-id': corpus_id,
                'score': score
            })
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Save results to TSV file
    results_path = os.path.join(output_dir, "results.tsv")
    results_df.to_csv(results_path, sep='\t', index=False)
    
    print(f"Results saved to {results_path}")