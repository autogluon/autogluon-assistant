from autogluon.assistant.tools_registry.indexing import TutorialIndexer

if __name__ == "__main__":
    # Initialize indexer
    indexer = TutorialIndexer()

    loaded_successfully = indexer.load_indices()
    if not loaded_successfully:
        # Build indices for all tools
        indexer.build_indices()
        # Save to disk
        indexer.save_indices()

    # Search for tutorials
    results = indexer.search(
        query="semantic segmentation",
        tool_name="autogluon.multimodal",
        condensed=True,
        top_k=5,
    )

    # Access full content
    for result in results:
        print(f"File: {result['file_path']}")
        print(f"Score: {result['score']}")
        print(f"Content: {result['content'][:200]}...")

    indexer.cleanup()
