# Condensed: ```python

Summary: This tutorial demonstrates implementing semantic search using AutoGluon's MultiModal (AutoMM) framework. It covers: (1) embedding extraction and semantic similarity computation using pre-trained language models, (2) implementing and evaluating BM25, pure semantic search, and hybrid search approaches, and (3) creating a ranking system with NDCG evaluation. Key functionalities include document embedding, semantic similarity calculation, and hybrid search combining lexical (BM25) and semantic signals. The tutorial helps with building efficient search systems, document retrieval, and implementing information retrieval evaluation metrics, showing how semantic search outperforms traditional keyword-based approaches.

*This is a condensed version that preserves essential implementation details and context.*

# Semantic Search with AutoMM

## Setup

```python
!pip install autogluon.multimodal
!pip3 install ir_datasets rank_bm25

import ir_datasets
import pandas as pd
import nltk
from collections import defaultdict
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils import compute_ranking_score, semantic_search, compute_semantic_similarity
import torch
```

## Dataset Preparation

```python
# Load NF Corpus dataset
dataset = ir_datasets.load("beir/nfcorpus/test")

# Prepare dataframes
doc_data = pd.DataFrame(dataset.docs_iter())
query_data = pd.DataFrame(dataset.queries_iter())
labeled_data = pd.DataFrame(dataset.qrels_iter())

# Define key columns
label_col = "relevance"
query_id_col = "query_id"
doc_id_col = "doc_id"
text_col = "text"

# Clean data
query_data = query_data.drop("url", axis=1)
doc_data[text_col] = doc_data[[text_col, "title"]].apply(" ".join, axis=1)
doc_data = doc_data.drop(["title", "url"], axis=1)

# Create id mappings for evaluation
id_mappings = {
    query_id_col: query_data.set_index(query_id_col)[text_col], 
    doc_id_col: doc_data.set_index(doc_id_col)[text_col]
}
```

## Evaluation Metric: NDCG

NDCG (Normalized Discounted Cumulative Gain) measures ranking performance with emphasis on top results:

- **DCG**: $\mathrm{DCG}_p = \sum_{i=1}^p \frac{\mathrm{rel}_i}{\log_2(i + 1)}$
- **NDCG**: $\mathrm{NDCG}_p = \frac{\mathrm{DCG}_p}{\mathrm{IDCG}_p}$

```python
cutoffs = [5, 10, 20]  # Evaluation at different cutoff points
```

## BM25 Implementation

```python
def tokenize_corpus(corpus):
    stop_words = set(stopwords.words("english") + list(string.punctuation))
    tokenized_docs = []
    for doc in corpus:
        tokens = nltk.word_tokenize(doc.lower())
        tokenized_doc = [w for w in tokens if w not in stop_words and len(w) > 2]
        tokenized_docs.append(tokenized_doc)
    return tokenized_docs

def rank_documents_bm25(queries_text, queries_id, docs_id, top_k, bm25):
    tokenized_queries = tokenize_corpus(queries_text)
    results = {qid: {} for qid in queries_id}
    for query_idx, query in enumerate(tokenized_queries):
        scores = bm25.get_scores(query)
        scores_top_k_idx = np.argsort(scores)[::-1][:top_k]
        for doc_idx in scores_top_k_idx:
            results[queries_id[query_idx]][docs_id[doc_idx]] = float(scores[doc_idx])
    return results

def get_qrels(dataset):
    qrel_dict = defaultdict(dict)
    for qrel in dataset.qrels_iter():
        qrel_dict[qrel.query_id][qrel.doc_id] = qrel.relevance
    return qrel_dict

def evaluate_bm25(doc_data, query_data, qrel_dict, cutoffs):
    tokenized_corpus = tokenize_corpus(doc_data[text_col].tolist())
    bm25_model = BM25Okapi(tokenized_corpus, k1=1.2, b=0.75)
    results = rank_documents_bm25(query_data[text_col].tolist(), query_data[query_id_col].tolist(), 
                                 doc_data[doc_id_col].tolist(), max(cutoffs), bm25_model)
    ndcg = compute_ranking_score(results=results, qrel_dict=qrel_dict, metrics=["ndcg"], cutoffs=cutoffs)
    return ndcg

# Get ground truth and evaluate BM25
qrel_dict = get_qrels(dataset)
bm25_scores = evaluate_bm25(doc_data, query_data, qrel_dict, cutoffs)
```

## AutoMM for Semantic Search

```python
# Initialize predictor
predictor = MultiModalPredictor(
    query=query_id_col,
    response=doc_id_col,
    label=label_col,
    problem_type="text_similarity",
    hyperparameters={"model.hf_text.checkpoint_name": "sentence-transformers/all-MiniLM-L6-v2"}
)

# Evaluate ranking performance
automm_scores = predictor.evaluate(
    labeled_data,
    query_data=query_data[[query_id_col]],
    response_data=doc_data[[doc_id_col]],
    id_mappings=id_mappings,
    cutoffs=cutoffs,
    metrics=["ndcg"],
)

# Perform semantic search
hits = semantic_search(
    matcher=predictor,
    query_data=query_data[text_col].tolist(),
    response_data=doc_data[text_col].tolist(),
    query_chunk_size=len(query_data),
    top_k=max(cutoffs),
)

# Extract embeddings
query_embeds = predictor.extract_embedding(query_data[[query_id_col]], id_mappings=id_mappings, as_tensor=True)
doc_embeds = predictor.extract_embedding(doc_data[[doc_id_col]], id_mappings=id_mappings, as_tensor=True)
```

## Hybrid BM25 Implementation

Combines BM25 and semantic embeddings with the formula:
`score = β * normalized_BM25 + (1 - β) * score_of_plm`

```python
def hybridBM25(query_data, query_embeds, doc_data, doc_embeds, recall_num, top_k, beta):
    # Recall documents with BM25 scores
    tokenized_corpus = tokenize_corpus(doc_data[text_col].tolist())
    bm25_model = BM25Okapi(tokenized_corpus, k1=1.2, b=0.75)
    bm25_scores = rank_documents_bm25(query_data[text_col].tolist(), query_data[query_id_col].tolist(), 
                                     doc_data[doc_id_col].tolist(), recall_num, bm25_model)
    
    # Normalize BM25 scores
    all_bm25_scores = [score for scores in bm25_scores.values() for score in scores.values()]
    max_bm25_score = max(all_bm25_scores)
    min_bm25_score = min(all_bm25_scores)

    # Prepare embeddings
    q_embeddings = {qid: embed for qid, embed in zip(query_data[query_id_col].tolist(), query_embeds)}
    d_embeddings = {did: embed for did, embed in zip(doc_data[doc_id_col].tolist(), doc_embeds)}
    
    # Calculate hybrid scores
    query_ids = query_data[query_id_col].tolist()
    results = {qid: {} for qid in query_ids}
    for idx, qid in enumerate(query_ids):
        rec_docs = bm25_scores[qid]
        rec_doc_emb = [d_embeddings[doc_id] for doc_id in rec_docs.keys()]
        rec_doc_id = [doc_id for doc_id in rec_docs.keys()]
        rec_doc_emb = torch.stack(rec_doc_emb)
        scores = compute_semantic_similarity(q_embeddings[qid], rec_doc_emb)
        scores[torch.isnan(scores)] = -1
        top_k_values, top_k_idxs = torch.topk(
            scores,
            min(top_k + 1, len(scores[0])),
            dim=1,
            largest=True,
            sorted=False,
        )

        for doc_idx, score in zip(top_k_idxs[0], top_k_values[0]):
            doc_id = rec_doc_id[int(doc_idx)]
            # Hybrid scores from BM25 and cosine similarity of embeddings
            normalized_bm25 = (bm25_scores[qid][doc_id] - min_bm25_score) / (max_bm25_score - min_bm25_score)
            results[qid][doc_id] = (1 - beta) * float(score.numpy()) + beta * normalized_bm25
    
    return results

def evaluate_hybridBM25(query_data, query_embeds, doc_data, doc_embeds, recall_num, beta, cutoffs):
    results = hybridBM25(query_data, query_embeds, doc_data, doc_embeds, recall_num, max(cutoffs), beta)
    ndcg = compute_ranking_score(results=results, qrel_dict=qrel_dict, metrics=["ndcg"], cutoffs=cutoffs)
    return ndcg

# Evaluate Hybrid BM25
recall_num = 1000
beta = 0.3
hybrid_scores = evaluate_hybridBM25(query_data, query_embeds, doc_data, doc_embeds, recall_num, beta, cutoffs)
```

## Key Findings

1. AutoMM significantly outperforms traditional BM25 for semantic search
2. Hybrid BM25 (combining BM25 with embedding similarity) provides further improvements
3. The embedding extraction capability enables efficient offline/online search systems