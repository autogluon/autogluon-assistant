# Condensed: Text Semantic Search with AutoMM

Summary: This tutorial demonstrates implementing semantic search using AutoGluon's MultiModalPredictor, covering both pure semantic search and hybrid approaches combining BM25. It provides code for text embedding generation, document ranking, and score normalization. Key implementations include data preprocessing, BM25 scoring, semantic embedding extraction using pre-trained transformers, and a hybrid scoring method that combines both approaches. The tutorial helps with tasks like document retrieval, similarity scoring, and ranking optimization. Notable features include configurable evaluation metrics (NDCG), customizable model parameters, and practical considerations for production deployment using efficient similarity search methods like Faiss.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on key implementation details and concepts:

# Text Semantic Search with AutoMM

## Key Concepts
- Semantic embedding converts text into feature vectors that encode semantic meaning
- Advantages over classical methods:
  - Matches by meaning rather than word usage
  - More computationally efficient
  - Supports multi-modal search

## Implementation

### 1. Setup and Data Preparation
```python
!pip install autogluon.multimodal ir_datasets
import ir_datasets
import pandas as pd

# Load dataset
dataset = ir_datasets.load("beir/nfcorpus/test")
doc_data = pd.DataFrame(dataset.docs_iter())
query_data = pd.DataFrame(dataset.queries_iter())
labeled_data = pd.DataFrame(dataset.qrels_iter())

# Data preprocessing
doc_data[text_col] = doc_data[[text_col, "title"]].apply(" ".join, axis=1)
doc_data = doc_data.drop(["title", "url"], axis=1)
```

### 2. BM25 Implementation
```python
from rank_bm25 import BM25Okapi

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
```

### 3. AutoMM Implementation
```python
from autogluon.multimodal import MultiModalPredictor

# Initialize predictor
predictor = MultiModalPredictor(
    query=query_id_col,
    response=doc_id_col,
    label=label_col,
    problem_type="text_similarity",
    hyperparameters={"model.hf_text.checkpoint_name": "sentence-transformers/all-MiniLM-L6-v2"}
)

# Evaluate ranking
results = predictor.evaluate(
    labeled_data,
    query_data=query_data[[query_id_col]],
    response_data=doc_data[[doc_id_col]],
    id_mappings=id_mappings,
    cutoffs=cutoffs,
    metrics=["ndcg"]
)

# Extract embeddings
query_embeds = predictor.extract_embedding(query_data[[query_id_col]], id_mappings=id_mappings, as_tensor=True)
doc_embeds = predictor.extract_embedding(doc_data[[doc_id_col]], id_mappings=id_mappings, as_tensor=True)
```

### 4. Hybrid BM25
```python
def hybridBM25(query_data, query_embeds, doc_data, doc_embeds, recall_num, top_k, beta):
    # Combine BM25 and semantic search scores
    bm25_scores = rank_documents_bm25(...)
    
    # Normalize BM25 scores
    all_bm25_scores = [score for scores in bm25_scores.values() for score in scores.values()]
    max_bm25_score = max(all_bm25_scores)
    min_bm25_score = min(all_bm25_scores)
    
    # Compute final scores
    results[qid][doc_id] = (1 - beta) * float(score.numpy()) + \
        beta * (bm25_scores[qid][doc_id] - min_bm25_score) / (max_bm25_score - min_bm25_score)
```

## Important Parameters
- `cutoffs`: List of positions to evaluate NDCG [5, 10, 20]
- `recall_num`: Number of documents to recall (1000)
- `beta`: Weight for hybrid scoring (0.3)
- BM25 parameters: k1=1.2, b=0.75

## Best Practices
1. Pre-compute and store document embeddings for efficiency
2. Use efficient similarity search methods (e.g., Faiss) for production
3. Consider hybrid approaches combining BM25 and semantic search
4. Normalize scores when combining different ranking methods