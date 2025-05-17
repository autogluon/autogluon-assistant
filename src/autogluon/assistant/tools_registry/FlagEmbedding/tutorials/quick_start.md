Summary: This tutorial demonstrates the implementation of a text retrieval system using BGE (BAAI/bge-base-en-v1.5) embeddings. It covers essential techniques for generating 768-dimensional text embeddings, computing similarity scores using dot product, and ranking results. The implementation includes specific code for model initialization with query instructions, FP16 optimization, embedding generation, and evaluation using Mean Reciprocal Rank (MRR). This tutorial is particularly useful for building semantic search applications, implementing text similarity comparisons, and setting up efficient retrieval systems. Key features include optimized model configuration, similarity calculation methods, and standardized evaluation metrics.

# Quick Start

In this tutorial, we will show how to use BGE models on a text retrieval task in 5 minutes.

## Step 0: Preparation

First, install FlagEmbedding in the environment.


```python
%pip install -U FlagEmbedding
```

Below is a super tiny courpus with only 10 sentences, which will be the dataset we use.

Each sentence is a concise discription of a famous people in specific domain.


```python
corpus = [
    "Michael Jackson was a legendary pop icon known for his record-breaking music and dance innovations.",
    "Fei-Fei Li is a professor in Stanford University, revolutionized computer vision with the ImageNet project.",
    "Brad Pitt is a versatile actor and producer known for his roles in films like 'Fight Club' and 'Once Upon a Time in Hollywood.'",
    "Geoffrey Hinton, as a foundational figure in AI, received Turing Award for his contribution in deep learning.",
    "Eminem is a renowned rapper and one of the best-selling music artists of all time.",
    "Taylor Swift is a Grammy-winning singer-songwriter known for her narrative-driven music.",
    "Sam Altman leads OpenAI as its CEO, with astonishing works of GPT series and pursuing safe and beneficial AI.",
    "Morgan Freeman is an acclaimed actor famous for his distinctive voice and diverse roles.",
    "Andrew Ng spread AI knowledge globally via public courses on Coursera and Stanford University.",
    "Robert Downey Jr. is an iconic actor best known for playing Iron Man in the Marvel Cinematic Universe.",
]
```

We want to know which one of these people could be an expert of neural network and who he/she is. 

Thus we generate the following query:


```python
query = "Who could be an expert of neural network?"
```

## Step 1: Text -> Embedding

First, let's use a [BGE embedding model](https://huggingface.co/BAAI/bge-base-en-v1.5) to create sentence embedding for the corpus.


```python
from FlagEmbedding import FlagModel

# get the BGE embedding model
model = FlagModel('BAAI/bge-base-en-v1.5',
                  query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                  use_fp16=True)

# get the embedding of the query and corpus
corpus_embeddings = model.encode(corpus)
query_embedding = model.encode(query)
```

The embedding of each sentence is a vector with length 768. 


```python
print("shape of the query embedding:  ", query_embedding.shape)
print("shape of the corpus embeddings:", corpus_embeddings.shape)
```

Run the following print line to take a look at the first 10 elements of the query embedding vector.


```python
print(query_embedding[:10])
```

## Step 2: Calculate Similarity

Now, we have the embeddings of the query and the corpus. The next step is to calculate the similarity between the query and each sentence in the corpus. Here we use the dot product/inner product as our similarity metric.


```python
sim_scores = query_embedding @ corpus_embeddings.T
print(sim_scores)
```

The result is a list of score representing the query's similarity to: [sentence 0, sentence 1, sentence 2, ...]

## Step 3: Ranking

After we have the similarity score of the query to each sentence in the corpus, we can rank them from large to small.


```python
# get the indices in sorted order
sorted_indices = sorted(range(len(sim_scores)), key=lambda k: sim_scores[k], reverse=True)
print(sorted_indices)
```

Now from the ranking, the sentence with index 3 is the best answer to our query "Who could be an expert of neural network?"

And that person is Geoffrey Hinton!


```python
print(corpus[3])
```

According to the order of indecies, we can print out the ranking of people that our little retriever got.


```python
# iteratively print the score and corresponding sentences in descending order

for i in sorted_indices:
    print(f"Score of {sim_scores[i]:.3f}: \"{corpus[i]}\"")
```

From the ranking, not surprisingly, the similarity scores of the query and the discriptions of Geoffrey Hinton and Fei-Fei Li is way higher than others, following by those of Andrew Ng and Sam Altman. 

While the key phrase "neural network" in the query does not appear in any of those discriptions, the BGE embedding model is still powerful enough to get the semantic meaning of query and corpus well.

## Step 4: Evaluate

We've seen the embedding model performed pretty well on the "neural network" query. What about the more general quality?

Let's generate a very small dataset of queries and corresponding ground truth answers. Note that the ground truth answers are the indices of sentences in the corpus.


```python
queries = [
    "Who could be an expert of neural network?",
    "Who might had won Grammy?",
    "Won Academy Awards",
    "One of the most famous female singers.",
    "Inventor of AlexNet",
]
```


```python
ground_truth = [
    [1, 3],
    [0, 4, 5],
    [2, 7, 9],
    [5],
    [3],
]
```

Here we repeat the steps we covered above to get the predicted ranking of each query.


```python
# use bge model to generate embeddings for all the queries
queries_embedding = model.encode(queries)
# compute similarity scores
scores = queries_embedding @ corpus_embeddings.T
# get he final rankings
rankings = [sorted(range(len(sim_scores)), key=lambda k: sim_scores[k], reverse=True) for sim_scores in scores]
rankings
```

Mean Reciprocal Rank ([MRR](https://en.wikipedia.org/wiki/Mean_reciprocal_rank)) is a widely used metric in information retrieval to evaluate the effectiveness of a system. Here we use that to have a very rough idea how our system performs.


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

We choose to use 1 and 5 as our cutoffs, with the result of 0.8 and 0.9 respectively.


```python
cutoffs = [1, 5]
mrrs = MRR(rankings, ground_truth, cutoffs)
for i, c in enumerate(cutoffs):
    print(f"MRR@{c}: {mrrs[i]}")
```
