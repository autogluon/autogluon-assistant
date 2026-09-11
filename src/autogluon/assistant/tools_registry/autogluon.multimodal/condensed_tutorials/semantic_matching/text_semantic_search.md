# Condensed: ```python

Summary: This tutorial demonstrates implementing semantic search and document ranking systems using AutoGluon's MultiModalPredictor with three approaches: classical BM25 (lexical matching with k1=1.2, b=0.75), transformer-based semantic search using sentence embeddings, and a hybrid method combining both. It covers essential tasks including: preparing IR datasets with query-document pairs, implementing BM25 with NLTK tokenization and rank_bm25, initializing MultiModalPredictor for text_similarity problems with sentence-transformers models, extracting embeddings for offline/online search architectures, performing semantic search with cosine similarity, evaluating ranking performance using NDCG metrics at multiple cutoffs, and building a two-stage hybrid retrieval system that recalls documents with BM25 then reranks using normalized weighted scores (β=0.3) from both lexical and semantic signals. Key utilities include compute_ranking_score, semantic_search, extract_embedding, and compute_semantic_similarity for production search engine implementations.

*This is a condensed version that preserves essential implementation details and context.*

# Semantic Search with AutoMM Tutorial (Condensed)

## Installation & Setup

```python
!pip install autogluon.multimodal
!pip3 install ir_datasets rank_bm25
import ir_datasets
import pandas as pd
```

## Dataset Preparation

Using NF Corpus (Nutrition Facts) from `ir_datasets`:

```python
dataset = ir_datasets.load("beir/nfcorpus/test")

# Prepare dataframes
doc_data = pd.DataFrame(dataset.docs_iter())
query_data = pd.DataFrame(dataset.queries_iter())
labeled_data = pd.DataFrame(dataset.qrels_iter())

# Define column names
label_col = "relevance"
query_id_col = "query_id"
doc_id_col = "doc_id"
text_col = "text"

# Create ID mappings
id_mappings = {
    query_id_col: query_data.set_index(query_id_col)[text_col], 
    doc_id_col: doc_data.set_index(doc_id_col)[text_col]
}

# Clean data
query_data = query_data.drop("url", axis=1)
doc_data[text_col] = doc_data[[text_col, "title"]].apply(" ".join, axis=1)
doc_data = doc_data.drop(["title", "url"], axis=1)
```

**Dataset size:** 323 queries, 3633 documents, 12334 relevance scores

## NDCG Evaluation Metric

**Key formulas:**

- **CG:** $\mathrm{CG}_p = \sum_{i=1}^p \mathrm{rel}_i$
- **DCG:** $\mathrm{DCG}_p = \sum_{i=1}^p \frac{\mathrm{rel}_i}{\log_2(i + 1)}$
- **IDCG:** $\mathrm{IDCG}_p = \sum_{i=1}^{|\mathrm{REL}_p|} \frac{\mathrm{rel}_i}{\log_2(i + 1)}$
- **NDCG:** $\mathrm{NDCG}_p = \frac{\mathrm{DCG}_p}{\mathrm{IDCG}_p}$

```python
from autogluon.multimodal.utils import compute_ranking_score
cutoffs = [5, 10, 20]
```

## BM25 Baseline

**BM25 Formula:**
$$score_{BM25} = \sum_i^n \mathrm{IDF}(q_i) \frac{f(q_i, D) \cdot (k1 + 1)}{f(q_i, D) + k1 \cdot (1 - b + b \cdot \frac{fieldLen}{avgFieldLen})}$$

**Default hyperparameters:** `k1=1.2`, `b=0.75`

```python
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def tokenize_corpus(corpus):
    stop_words = set(stopwords.words("english") + list(string.punctuation))
    tokenized_docs = []
    for doc in corpus:
        tokens = nltk.word_tokenize(doc.lower())
        tokenized_doc = [w for w in tokens if w not in stop_words and len(w) > 2]
        tokenized_docs.append(tokenized_doc)
    return tokenized_docs

def evaluate_bm25(doc_data, query_data, qrel_dict, cutoffs):
    tokenized_corpus = tokenize_corpus(doc_data[text_col].tolist())
    bm25_model = BM25Okapi(tokenized_corpus, k1=1.2, b=0.75)
    
    results = rank_documents_bm25(query_data[text_col].tolist(), 
                                  query_data[query_id_col].tolist(), 
                                  doc_data[doc_id_col].tolist(), 
                                  max(cutoffs), bm25_model)
    ndcg = compute_ranking_score(results=results, qrel_dict=qrel_dict, 
                                 metrics=["ndcg"], cutoffs=cutoffs)
    return ndcg
```

## AutoMM for Semantic Search

### Initialize Predictor

```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(
    query=query_id_col,
    response=doc_id_col,
    label=label_col,
    problem_type="text_similarity",
    hyperparameters={"model.hf_text.checkpoint_name": "sentence-transformers/all-MiniLM-L6-v2"}
)
```

### Evaluate Ranking

```python
predictor.evaluate(
    labeled_data,
    query_data=query_data[[query_id_col]],
    response_data=doc_data[[doc_id_col]],
    id_mappings=id_mappings,
    cutoffs=cutoffs,
    metrics=["ndcg"],
)
```

**Result:** Significant improvements over BM25

### Semantic Search

```python
from autogluon.multimodal.utils import semantic_search

hits = semantic_search(
    matcher=predictor,
    query_data=query_data[text_col].tolist(),
    response_data=doc_data[text_col].tolist(),
    query_chunk_size=len(query_data),
    top_k=max(cutoffs),
)
```

**Note:** Uses `torch.topk` with O(n+k) complexity. For production, consider [Faiss](https://github.com/facebookresearch/faiss) for efficient similarity search.

### Extract Embeddings

```python
query_embeds = predictor.extract_embedding(
    query_data[[query_id_col]], id_mappings=id_mappings, as_tensor=True
)
doc_embeds = predictor.extract_embedding(
    doc_data[[doc_id_col]], id_mappings=id_mappings, as_tensor=True
)
```

**Use case:** Extract database embeddings offline; encode queries online for efficient matching.

## Hybrid BM25 Method

Combines BM25 and semantic embeddings for improved ranking.

**Scoring formula:**
$$score = \beta \times normalized\_BM25 + (1 - \beta) \times score\_of\_plm$$

where:
$$normalized\_BM25(q_i, D_j) = \frac{\textsf{BM25}(q_i,D_j) - \min(\textsf{BM25})}{\max(\textsf{BM25}) - \min(\textsf{BM25})}$$

**Default:** `β = 0.3`, `recall_num = 1000`

```python
from autogluon.multimodal.utils import compute_semantic_similarity

def hybridBM25(query_data, query_embeds, doc_data, doc_embeds, recall_num, top_k, beta):
    # Stage 1: BM25 recall
    tokenized_corpus = tokenize_corpus(doc_data[text_col].tolist())
    bm25_model = BM25Okapi(tokenized_corpus, k1=1.2, b=0.75)
    bm25_scores = rank_documents_bm25(...)
    
    # Normalize BM25 scores
    all_bm25_scores = [score for scores in bm25_scores.values() for score in scores.values()]
    max_bm25_score = max(all_bm25_scores)
    min_bm25_score = min(all_bm25_scores)
    
    # Stage 2: Rerank with embeddings
    results = {qid: {} for qid in query_ids}
    for qid in query_ids:
        rec_docs = bm25_scores[qid]
        rec_doc_emb = torch.stack([d_embeddings[doc_id] for doc_id in rec_docs.keys()])
        scores = compute_semantic_similarity(q_embeddings[qid], rec_doc_emb)
        
        # Hybrid scoring
        for doc_id, plm_score in zip(doc_ids, plm_scores):
            results[qid][doc_id] = \
                (1 - beta) * float(plm_score) + \
                beta * (bm25_scores[qid][doc_id] - min_bm25_score) / (max_bm25_score - min_bm25_score)
    
    return results

# Evaluate
recall_num = 1000
beta = 0.3
evaluate_hybridBM25(query_data, query_embeds, doc_data, doc_embeds, recall_num, beta, cutoffs)
```

**Result:** Further improvements over both BM25 and AutoMM alone.

## Summary

- **AutoMM** provides easy APIs for semantic search with significant improvements over classical BM25
- **Hybrid BM25** combines lexical (BM25) and semantic (embeddings) matching for best results
- **Key methods:** `evaluate()`, `semantic_search()`, `extract_embedding()`
- **Production considerations:** Use efficient similarity search (e.g., Faiss) for large-scale deployment