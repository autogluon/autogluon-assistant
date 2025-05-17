# Condensed: Quick Start

Summary: This tutorial demonstrates the implementation of a text retrieval system using BGE (BAAI/bge-base-en-v1.5) embeddings. It covers essential techniques for generating 768-dimensional text embeddings, computing similarity scores using dot product, and ranking results. The implementation includes specific code for model initialization with query instructions, FP16 optimization, embedding generation, and evaluation using Mean Reciprocal Rank (MRR). This tutorial is particularly useful for building semantic search applications, implementing text similarity comparisons, and setting up efficient retrieval systems. Key features include optimized model configuration, similarity calculation methods, and standardized evaluation metrics.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# BGE Models Quick Start for Text Retrieval

## Key Implementation Steps

### 1. Setup
```python
from FlagEmbedding import FlagModel
%pip install -U FlagEmbedding
```

### 2. Initialize Model
```python
model = FlagModel('BAAI/bge-base-en-v1.5',
                  query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                  use_fp16=True)
```

### 3. Generate Embeddings
```python
# Convert text to embeddings (768-dimensional vectors)
corpus_embeddings = model.encode(corpus)
query_embedding = model.encode(query)
```

### 4. Calculate Similarity & Ranking
```python
# Compute similarity scores using dot product
sim_scores = query_embedding @ corpus_embeddings.T

# Rank results
sorted_indices = sorted(range(len(sim_scores)), 
                       key=lambda k: sim_scores[k], 
                       reverse=True)
```

### 5. Evaluation Using MRR
```python
def MRR(preds, labels, cutoffs):
    mrr = [0 for _ in range(len(cutoffs))]
    for pred, label in zip(preds, labels):
        for i, c in enumerate(cutoffs):
            for j, index in enumerate(pred):
                if j < c and index in label:
                    mrr[i] += 1/(j+1)
                    break
    mrr = [k/len(preds) for k in mrr]
    return mrr
```

## Critical Configurations

- Model: `BAAI/bge-base-en-v1.5`
- Embedding dimension: 768
- Query instruction prefix: "Represent this sentence for searching relevant passages:"
- FP16 optimization enabled

## Best Practices

1. Use appropriate query instructions for retrieval tasks
2. Enable FP16 for better performance
3. Use dot product for similarity calculation
4. Evaluate using standard metrics like MRR with appropriate cutoffs

## Important Notes

- The model can understand semantic relationships even without exact keyword matches
- Embeddings are 768-dimensional vectors suitable for similarity comparisons
- The system can be evaluated using standard IR metrics like MRR@1 and MRR@5

This implementation provides a basic but effective text retrieval system using BGE embeddings, suitable for semantic search applications.