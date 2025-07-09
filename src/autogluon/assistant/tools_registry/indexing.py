import contextlib
import io
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from FlagEmbedding import FlagAutoModel

from .registry import ToolsRegistry

logger = logging.getLogger(__name__)

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"


class TutorialIndexer:
    """
    Indexes tutorial summaries using FAISS and BGE embeddings for efficient retrieval.
    Maintains separate indices for regular and condensed tutorials for each tool.
    """

    def __init__(self, embedding_model_name: str = "BAAI/bge-base-en-v1.5"):
        self.registry = ToolsRegistry()
        self.embedding_model_name = embedding_model_name
        self.sanitized_model_name = self.embedding_model_name.replace("/", "_")
        self.model = None
        self.indices: Dict[str, Dict[str, faiss.Index]] = {}  # {tool_name: {type: index}}
        self.metadata: Dict[str, Dict[str, List[Dict]]] = {}  # {tool_name: {type: [metadata]}}
        self.index_dir = Path(__file__).parent / "indices" / self.sanitized_model_name
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def __del__(self):
        """Cleanup method to properly close the embedding model."""
        self.cleanup()

    def __silent_encode(self, input):
        with contextlib.redirect_stderr(io.StringIO()):
            return self.model.encode(input)

    def cleanup(self):
        """Cleanup the embedding model to avoid multiprocessing issues."""
        if self.model is not None:
            try:
                # Try to properly close the model if it has cleanup methods
                if hasattr(self.model, "close"):
                    self.model.close()
                elif hasattr(self.model, "stop_multi_process_pool"):
                    self.model.stop_multi_process_pool()
                else:
                    del self.model
            except Exception as e:
                logger.debug(f"Error during model cleanup: {e}")
            finally:
                self.model = None

    def _load_embedding_model(self):
        """Load the BGE embedding model lazily."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.model = FlagAutoModel.from_finetuned(
                self.embedding_model_name,
                query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                use_fp16=True,
                devices=None,  # Use single process mode
                batch_size=32,  # Reasonable batch size
                normalize_embeddings=False,  # We'll handle normalization ourselves
            )
            logger.info("Embedding model loaded successfully")

    def _extract_summary_from_md(self, md_path: Path) -> Optional[str]:
        """
        Extract summary from markdown file.

        Args:
            md_path: Path to markdown file

        Returns:
            Summary text if found, None otherwise
        """
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Look for summary line
            lines = content.split("\n")
            for line in lines:
                if line.strip().startswith("Summary: "):
                    return line.strip()[9:]  # Remove "Summary: " prefix

            return None
        except Exception as e:
            logger.warning(f"Error extracting summary from {md_path}: {e}")
            return None

    def _build_tool_index(self, tool_name: str, tutorial_type: str) -> Tuple[faiss.Index, List[Dict]]:
        """
        Build FAISS index for a specific tool and tutorial type.

        Args:
            tool_name: Name of the tool
            tutorial_type: Either 'tutorials' or 'condensed_tutorials'

        Returns:
            Tuple of (FAISS index, metadata list)
        """
        self._load_embedding_model()

        try:
            if tutorial_type == "condensed_tutorials":
                tutorials_folder = self.registry.get_tool_tutorials_folder(tool_name, condensed=True)
            else:
                tutorials_folder = self.registry.get_tool_tutorials_folder(tool_name, condensed=False)
        except FileNotFoundError as e:
            logger.warning(f"No {tutorial_type} found for tool {tool_name}: {e}")
            return None, []

        summaries = []
        metadata = []

        # Recursively find all .md files
        for md_file in tutorials_folder.rglob("*.md"):
            summary = self._extract_summary_from_md(md_file)
            if summary and summary.strip():  # Ensure non-empty summary
                summaries.append(summary)
                metadata.append(
                    {
                        "tool_name": tool_name,
                        "tutorial_type": tutorial_type,
                        "file_path": str(md_file),
                        "relative_path": str(md_file.relative_to(tutorials_folder)),
                        "summary": summary,
                    }
                )

        if not summaries:
            logger.warning(f"No summaries found for {tool_name} {tutorial_type}")
            return None, []

        logger.info(f"Found {len(summaries)} summaries for {tool_name} {tutorial_type}")

        try:
            # Generate embeddings
            logger.info(f"Generating embeddings for {tool_name} {tutorial_type}")

            # Process in smaller batches to avoid memory issues
            batch_size = 16
            all_embeddings = []

            for i in range(0, len(summaries), batch_size):
                batch_summaries = summaries[i : i + batch_size]
                batch_embeddings = self.__silent_encode(batch_summaries)

                # Ensure proper format
                if not isinstance(batch_embeddings, np.ndarray):
                    batch_embeddings = np.array(batch_embeddings)

                all_embeddings.append(batch_embeddings)

            # Concatenate all embeddings
            embeddings = np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]

            # Convert to float32 and ensure contiguous memory layout
            embeddings = np.ascontiguousarray(embeddings.astype(np.float32))

            # Validate embeddings shape and values
            if len(embeddings.shape) != 2:
                raise ValueError(f"Expected 2D embeddings array, got shape {embeddings.shape}")

            if embeddings.size == 0:
                raise ValueError("Embeddings array is empty")

            # Check for NaN or infinite values
            if not np.isfinite(embeddings).all():
                logger.warning(f"Found non-finite values in embeddings for {tool_name} {tutorial_type}")
                # Replace NaN/inf with zeros
                embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

            # Create FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Inner product for similarity

            # Normalize embeddings for cosine similarity (in-place)
            # Make a copy to avoid modifying the original array
            embeddings_normalized = embeddings.copy()
            faiss.normalize_L2(embeddings_normalized)

            # Add embeddings to index
            index.add(embeddings_normalized)

            logger.info(f"Built index for {tool_name} {tutorial_type} with {index.ntotal} vectors")
            return index, metadata

        except Exception as e:
            logger.error(f"Error during embedding generation or indexing for {tool_name} {tutorial_type}: {e}")
            return None, []

    def build_indices(self, tools: Optional[List[str]] = None) -> None:
        """
        Build FAISS indices for all tools or specified tools.

        Args:
            tools: List of tool names to index. If None, index all tools.
        """
        if tools is None:
            tools = self.registry.list_tools()

        logger.info(f"Building indices for tools: {tools}")

        for tool_name in tools:
            logger.info(f"Processing tool: {tool_name}")

            # Initialize tool dictionaries
            if tool_name not in self.indices:
                self.indices[tool_name] = {}
                self.metadata[tool_name] = {}

            # Build indices for both tutorial types
            for tutorial_type in ["tutorials", "condensed_tutorials"]:
                try:
                    index, metadata = self._build_tool_index(tool_name, tutorial_type)

                    if index is not None:
                        self.indices[tool_name][tutorial_type] = index
                        self.metadata[tool_name][tutorial_type] = metadata
                        logger.info(f"Successfully built {tutorial_type} index for {tool_name}")
                    else:
                        logger.warning(f"Failed to build {tutorial_type} index for {tool_name}")

                except Exception as e:
                    logger.error(f"Error building {tutorial_type} index for {tool_name}: {e}")
                    continue

    def save_indices(self) -> None:
        """Save all indices and metadata to disk."""
        logger.info("Saving indices to disk")

        for tool_name, tool_indices in self.indices.items():
            tool_dir = self.index_dir / tool_name
            tool_dir.mkdir(exist_ok=True)

            for tutorial_type, index in tool_indices.items():
                # Save FAISS index
                index_file = tool_dir / f"{tutorial_type}.index"
                faiss.write_index(index, str(index_file))

                # Save metadata
                metadata_file = tool_dir / f"{tutorial_type}.metadata"
                with open(metadata_file, "wb") as f:
                    pickle.dump(self.metadata[tool_name][tutorial_type], f)

                logger.info(f"Saved {tool_name} {tutorial_type} index and metadata")

    def load_indices(self) -> None:
        """Load all indices and metadata from disk."""
        logger.info("Loading indices from disk")

        self.indices = {}
        self.metadata = {}

        # Check if index_dir is empty
        if not any(self.index_dir.iterdir()):
            logger.warning(f"No index exists at {self.index_dir}")
            return False

        for tool_dir in self.index_dir.iterdir():
            if tool_dir.is_dir():
                tool_name = tool_dir.name
                self.indices[tool_name] = {}
                self.metadata[tool_name] = {}

                for tutorial_type in ["tutorials", "condensed_tutorials"]:
                    index_file = tool_dir / f"{tutorial_type}.index"
                    metadata_file = tool_dir / f"{tutorial_type}.metadata"

                    if index_file.exists() and metadata_file.exists():
                        try:
                            # Load FAISS index
                            index = faiss.read_index(str(index_file))
                            self.indices[tool_name][tutorial_type] = index

                            # Load metadata
                            with open(metadata_file, "rb") as f:
                                metadata = pickle.load(f)
                            self.metadata[tool_name][tutorial_type] = metadata

                            logger.info(f"Loaded {tool_name} {tutorial_type} index with {index.ntotal} vectors")

                        except Exception as e:
                            logger.error(f"Error loading {tool_name} {tutorial_type} index: {e}")
                            return False
                    else:
                        logger.error(f"There is no {index_file} or {metadata_file}.")
                        return False
        return True

    def search(self, query: str, tool_name: str, condensed: bool = False, top_k: int = 5) -> List[Dict]:
        """
        Search for relevant tutorials based on query.

        Args:
            query: Search query
            tool_name: Name of the tool to search in
            condensed: Whether to search in condensed tutorials
            top_k: Number of top results to return

        Returns:
            List of dictionaries containing tutorial information and content
        """
        self._load_embedding_model()

        tutorial_type = "condensed_tutorials" if condensed else "tutorials"

        # Check if index exists
        if tool_name not in self.indices or tutorial_type not in self.indices[tool_name]:
            logger.warning(f"No index found for {tool_name} {tutorial_type}")
            return []

        index = self.indices[tool_name][tutorial_type]
        metadata = self.metadata[tool_name][tutorial_type]

        if index.ntotal == 0:
            logger.warning(f"Empty index for {tool_name} {tutorial_type}")
            return []

        # Generate query embedding
        query_embedding = self.__silent_encode([query])

        # Ensure proper data type and memory layout
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)
        query_embedding = np.ascontiguousarray(query_embedding.astype(np.float32))

        # Normalize query embedding
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # No more results
                break

            meta = metadata[idx]

            # Load full content from file
            try:
                with open(meta["file_path"], "r", encoding="utf-8") as f:
                    content = f.read()

                result = {
                    "tool_name": meta["tool_name"],
                    "tutorial_type": meta["tutorial_type"],
                    "file_path": meta["file_path"],
                    "relative_path": meta["relative_path"],
                    "summary": meta["summary"],
                    "content": content,
                    "score": float(score),
                }
                results.append(result)

            except Exception as e:
                logger.error(f"Error loading content from {meta['file_path']}: {e}")
                continue

        logger.info(f"Found {len(results)} results for query: {query}")
        return results

    def get_all_summaries(self, tool_name: str, condensed: bool = False) -> List[Dict]:
        """
        Get all summaries for a specific tool and tutorial type.

        Args:
            tool_name: Name of the tool
            condensed: Whether to get condensed tutorials

        Returns:
            List of summary dictionaries
        """
        tutorial_type = "condensed_tutorials" if condensed else "tutorials"

        if tool_name not in self.metadata or tutorial_type not in self.metadata[tool_name]:
            return []

        return self.metadata[tool_name][tutorial_type]

    def rebuild_tool_index(self, tool_name: str) -> None:
        """
        Rebuild indices for a specific tool.

        Args:
            tool_name: Name of the tool to rebuild
        """
        logger.info(f"Rebuilding indices for tool: {tool_name}")
        self.build_indices([tool_name])

    def delete_tool_indices(self, tool_name: str) -> None:
        """
        Delete indices for a specific tool.

        Args:
            tool_name: Name of the tool to delete
        """
        # Remove from memory
        if tool_name in self.indices:
            del self.indices[tool_name]
        if tool_name in self.metadata:
            del self.metadata[tool_name]

        # Remove from disk
        tool_dir = self.index_dir / tool_name
        if tool_dir.exists():
            import shutil

            shutil.rmtree(tool_dir)
            logger.info(f"Deleted indices for tool: {tool_name}")

    def get_index_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get statistics about all indices.

        Returns:
            Dictionary with index statistics
        """
        stats = {}
        for tool_name, tool_indices in self.indices.items():
            stats[tool_name] = {}
            for tutorial_type, index in tool_indices.items():
                stats[tool_name][tutorial_type] = index.ntotal
        return stats

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
