import random
import os
import pandas as pd

def process_corpus(corpus_path, gt_path, output_path):
    """
    Process a corpus to create a reduced version that keeps 10% of documents
    and ensures all documents referenced in the ground truth are included.
    
    Args:
        corpus_path (str): Path to the corpus JSONL file
        gt_path (str): Path to the ground truth/queries file
        output_path (str): Path to save the reduced corpus
    """
    # Load the ground truth TSV file using pandas
    try:
        gt_df = pd.read_csv(gt_path, sep='\t')
        # Extract the corpus-id column (second column in TSV)
        ground_truth_docs = set(gt_df['corpus-id'])
        print(f"Successfully read ground truth TSV from {gt_path}")
    except Exception as e:
        print(f"Error reading ground truth TSV file {gt_path}: {e}")
        ground_truth_docs = set()
    
    print(f"Number of documents in ground truth: {len(ground_truth_docs)}")
    
    # Read the corpus using pandas
    try:
        corpus_df = pd.read_json(corpus_path, lines=True)
        corpus_docs = corpus_df.to_dict('records')
    except Exception as e:
        print(f"Error reading corpus file {corpus_path}: {e}")
        corpus_docs = []
    
    print(f"Total corpus size: {len(corpus_docs)} documents")
    
    # Calculate how many documents to keep (10% of total)
    num_to_keep = max(int(len(corpus_docs) * 0.1), len(ground_truth_docs))
    
    # Create a set of document IDs from corpus that match ground truth
    ground_truth_id_set = set()
    id_field = '_id'  # Default ID field name
    
    # First, detect which field contains the document ID
    if len(corpus_docs) > 0:
        sample_doc = corpus_docs[0]
        potential_id_fields = ['_id', 'id', 'doc_id', 'document_id']
        for field in potential_id_fields:
            if field in sample_doc:
                id_field = field
                break
    
    # Now collect documents that match ground truth IDs
    for doc in corpus_docs:
        doc_id = doc.get(id_field)
        if doc_id in ground_truth_docs:
            ground_truth_id_set.add(doc_id)
    
    # Identify which ground truth documents are missing from corpus
    missing_docs = ground_truth_docs - ground_truth_id_set
    if missing_docs:
        print(f"Warning: {len(missing_docs)} documents from ground truth not found in corpus")
        if len(missing_docs) < 10:  # Only print if there aren't too many
            print(f"Missing documents: {missing_docs}")
    
    # Separate documents into ground truth docs and other docs
    ground_truth_documents = []
    other_documents = []
    
    for doc in corpus_docs:
        doc_id = doc.get(id_field)
        if doc_id in ground_truth_docs:
            ground_truth_documents.append(doc)
        else:
            other_documents.append(doc)
    
    # Calculate how many additional docs we need beyond ground truth docs
    additional_needed = num_to_keep - len(ground_truth_documents)
    additional_needed = max(0, additional_needed)  # Ensure non-negative
    
    # Select random subset of other documents
    selected_others = []
    if additional_needed > 0 and other_documents:
        selected_others = random.sample(other_documents, min(additional_needed, len(other_documents)))
    
    # Combine ground truth documents with randomly selected others
    final_corpus = ground_truth_documents + selected_others
    
    print(f"Final corpus size: {len(final_corpus)} documents")
    print(f"Ground truth documents included: {len(ground_truth_documents)} of {len(ground_truth_docs)}")
    print(f"Random additional documents: {len(selected_others)}")
    
    # Create parent directories if they don't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write the reduced corpus to the output file using pandas
    reduced_df = pd.DataFrame(final_corpus)
    reduced_df.to_json(output_path, orient='records', lines=True)
    
    print(f"Reduced corpus written to '{output_path}'")

if __name__ == "__main__":
    # Use the specified paths
    corpus_path = "/media/agent/maab/datasets/climate_fever/training/corpus.jsonl"
    gt_path = "/media/agent/maab/datasets/climate_fever/eval/ground_truth.tsv"
    output_path = "/media/agent/maab/datasets/climate_fever/training/reduced_corpus.jsonl"
    
    process_corpus(corpus_path, gt_path, output_path)
    