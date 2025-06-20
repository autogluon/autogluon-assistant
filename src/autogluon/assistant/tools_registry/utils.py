import json
import re
import time
from collections import defaultdict
from typing import Dict, List
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


def split_markdown_into_chunks(content: str, max_chunk_size: int = 4000) -> List[str]:
    """
    Split markdown content into chunks at logical boundaries.

    Args:
        content: The markdown content to split
        max_chunk_size: Maximum size of each chunk

    Returns:
        List of markdown chunks
    """
    # Split content into sections at header boundaries
    sections = []
    current_section = []
    for line in content.split("\n"):
        if line.startswith("#") and current_section:
            sections.append("\n".join(current_section))
            current_section = []
        current_section.append(line)
    if current_section:
        sections.append("\n".join(current_section))

    chunks = []
    current_chunk = []
    current_size = 0

    for section in sections:
        # If a single section is larger than max_chunk_size, split it into smaller pieces
        if len(section) > max_chunk_size:
            sub_chunks = _split_large_section(section, max_chunk_size)
            for sub_chunk in sub_chunks:
                chunks.append(sub_chunk)
            continue

        # If adding this section would exceed max_chunk_size, start a new chunk
        if current_size + len(section) > max_chunk_size and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_size = 0

        current_chunk.append(section)
        current_size += len(section)

    # Add any remaining content
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def _split_large_section(section: str, max_chunk_size: int) -> List[str]:
    """
    Split a large section into smaller chunks while preserving code blocks and paragraphs.

    Args:
        section: The section content to split
        max_chunk_size: Maximum size of each chunk

    Returns:
        List of section chunks
    """
    chunks = []
    lines = section.split("\n")
    current_chunk = []
    current_size = 0
    in_code_block = False
    code_block_content = []

    for line in lines:
        # Handle code blocks
        if line.startswith("```"):
            if in_code_block:
                # End of code block
                code_block_content.append(line)
                code_block = "\n".join(code_block_content)

                # If code block would exceed chunk size on its own, make it its own chunk
                if len(code_block) > max_chunk_size:
                    if current_chunk:
                        chunks.append("\n".join(current_chunk))
                        current_chunk = []
                    chunks.append(code_block)
                    current_size = 0
                else:
                    # If adding code block would exceed size, start new chunk
                    if current_size + len(code_block) > max_chunk_size and current_chunk:
                        chunks.append("\n".join(current_chunk))
                        current_chunk = []
                        current_size = 0
                    current_chunk.extend(code_block_content)
                    current_size += len(code_block)

                code_block_content = []
                in_code_block = False
            else:
                # Start of code block
                in_code_block = True
                code_block_content = [line]
            continue

        if in_code_block:
            code_block_content.append(line)
            continue

        # Handle regular lines
        line_length = len(line) + 1  # +1 for newline

        # Start new chunk if adding this line would exceed max_size
        if current_size + line_length > max_chunk_size and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_size = 0

        current_chunk.append(line)
        current_size += line_length

    # Add any remaining content
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


class HuggingFaceModelsFetcher:
    def __init__(self):
        self.base_url = "https://huggingface.co/api"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "HF-Models-Fetcher/1.0"})

    def get_models_by_task(self, task: str, sort_by: str = "likes", limit: int = 10) -> List[Dict]:
        """Get models filtered by specific task"""
        url = f"{self.base_url}/models"
        params = {"pipeline_tag": task, "sort": sort_by, "limit": limit, "full": "true"}

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching models for task {task}: {e}")
            return []

    def get_all_pipeline_tags(self) -> List[str]:
        """Get all available pipeline tags/tasks from Hugging Face"""
        return ["any-to-any", "audio-text-to-text"]
        """
        return [
            # Multimodal
            "any-to-any",
            "audio-text-to-text",
            "document-question-answering",
            "visual-document-retrieval",
            "image-text-to-text",
            "video-text-to-text",
            "visual-question-answering",
            # Natural Language Processing
            "feature-extraction",
            "fill-mask",
            "question-answering",
            "sentence-similarity",
            "summarization",
            "table-question-answering",
            "text-classification",
            "text-generation",
            "text-ranking",
            "token-classification",
            "translation",
            "zero-shot-classification",
            # Computer Vision
            "depth-estimation",
            "image-classification",
            "image-feature-extraction",
            "image-segmentation",
            "image-to-image",
            "image-to-text",
            "image-to-video",
            "keypoint-detection",
            "mask-generation",
            "object-detection",
            "video-classification",
            "text-to-image",
            "text-to-video",
            "unconditional-image-generation",
            "zero-shot-image-classification",
            "zero-shot-object-detection",
            "text-to-3d",
            "image-to-3d",
            # Audio
            "audio-classification",
            "audio-to-audio",
            "automatic-speech-recognition",
            "text-to-speech",
            # Tabular
            "tabular-classification",
            "tabular-regression",
            # Reinforcement Learning
            "reinforcement-learning",
            # Additional common tasks that might use different naming conventions
            "conversational",
            "text2text-generation",
            "voice-activity-detection",
            "time-series-forecasting",
            "robotics",
            "other",
        ]
        """

    def extract_model_info(self, model: Dict) -> Dict:
        """Extract relevant information from model data"""
        model_id = model.get("id", "")
        return {
            "model_id": model_id,
            "url": f"https://huggingface.co/{model_id}" if model_id else "",
            "author": model.get("author", ""),
            "likes": model.get("likes", 0),
            "downloads": model.get("downloads", 0),
            "created_at": model.get("createdAt", ""),
            "last_modified": model.get("lastModified", ""),
            "pipeline_tag": model.get("pipeline_tag", ""),
            "library_name": model.get("library_name", ""),
            "tags": model.get("tags", []),
            "model_size": model.get("safetensors", {}).get("total", 0) if model.get("safetensors") else 0,
        }

    def get_top_models_all_tasks(
        self, n: int = 10, sort_by: str = "likes", include_downloads: bool = True
    ) -> Dict[str, List[Dict]]:
        """
        Get top N models for each task

        Args:
            n: Number of top models to fetch per task
            sort_by: Sort criteria ('likes', 'downloads', 'modified', 'created')
            include_downloads: Whether to also fetch top downloaded models

        Returns:
            Dictionary with task names as keys and list of model info as values
        """
        all_tasks_models = defaultdict(list)
        pipeline_tags = self.get_all_pipeline_tags()

        print(f"Fetching top {n} models for {len(pipeline_tags)} tasks...")

        for i, task in enumerate(pipeline_tags):
            print(f"Processing task {i+1}/{len(pipeline_tags)}: {task}")

            # Get top models by specified criteria (likes by default)
            models = self.get_models_by_task(task, sort_by=sort_by, limit=n)

            if models:
                task_models = []
                for model in models[:n]:  # Ensure we only get top N
                    model_info = self.extract_model_info(model)
                    model_info["sort_criteria"] = sort_by
                    task_models.append(model_info)

                all_tasks_models[task] = task_models
                print(f"  Found {len(task_models)} models for {task}")
            else:
                print(f"  No models found for {task}")

            # Add small delay to be respectful to the API
            time.sleep(0.1)

        return dict(all_tasks_models)

    def merge_top_models(
        self, liked_models: Dict[str, List[Dict]], downloaded_models: Dict[str, List[Dict]], max_per_task: int = 15
    ) -> Dict[str, List[Dict]]:
        """
        Merge liked and downloaded models, removing duplicates and keeping top models

        Args:
            liked_models: Dictionary of models sorted by likes
            downloaded_models: Dictionary of models sorted by downloads
            max_per_task: Maximum number of models to keep per task

        Returns:
            Dictionary of merged top models per task
        """
        merged_models = defaultdict(list)

        # Get all unique tasks
        all_tasks = set(liked_models.keys()) | set(downloaded_models.keys())

        for task in all_tasks:
            liked_list = liked_models.get(task, [])
            downloaded_list = downloaded_models.get(task, [])

            # Create a dictionary to track unique models by model_id
            unique_models = {}

            # Add liked models first
            for model in liked_list:
                model_id = model["model_id"]
                if model_id not in unique_models:
                    model_copy = model.copy()
                    model_copy["source"] = "liked"
                    unique_models[model_id] = model_copy

            # Add downloaded models, updating existing entries
            for model in downloaded_list:
                model_id = model["model_id"]
                if model_id in unique_models:
                    # Model exists in both lists, mark as both
                    unique_models[model_id]["source"] = "both"
                else:
                    # New model from downloads
                    model_copy = model.copy()
                    model_copy["source"] = "downloaded"
                    unique_models[model_id] = model_copy

            # Convert back to list and sort by a composite score
            models_list = list(unique_models.values())

            # Sort by composite score: prioritize models that appear in both lists,
            # then by likes + normalized downloads
            def composite_score(model):
                base_score = model.get("likes", 0) + (model.get("downloads", 0) / 1000)  # Normalize downloads
                if model["source"] == "both":
                    base_score *= 1.5  # Bonus for appearing in both lists
                return base_score

            models_list.sort(key=composite_score, reverse=True)

            # Keep only top models per task
            merged_models[task] = models_list[:max_per_task]

            print(f"Task {task}: {len(models_list)} unique models merged, keeping top {len(merged_models[task])}")

        return dict(merged_models)

    def save_to_json(self, data: Dict, filename: str = "top_hf_models.json"):
        """Save results to JSON file"""
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {filename}")
        except Exception as e:
            print(f"Error saving to JSON: {e}")

    def save_to_csv(self, data: Dict, filename: str = "top_hf_models.csv"):
        """Save results to CSV file"""
        try:
            # Flatten the data for CSV format
            rows = []
            for task, models in data.items():
                for model in models:
                    row = model.copy()
                    row["task"] = task
                    rows.append(row)

            df = pd.DataFrame(rows)
            df.to_csv(filename, index=False)
            print(f"Results saved to {filename}")
        except Exception as e:
            print(f"Error saving to CSV: {e}")

    def print_summary(self, data: Dict, title: str = "TOP MODELS BY TASK"):
        """Print a summary of the results"""
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)

        total_models = sum(len(models) for models in data.values())
        print(f"Total tasks processed: {len(data)}")
        print(f"Total models found: {total_models}")

        print("\nTop 3 most liked models overall:")
        all_models = []
        for task, models in data.items():
            for model in models:
                model_copy = model.copy()
                model_copy["task"] = task
                all_models.append(model_copy)

        # Sort by likes
        top_overall = sorted(all_models, key=lambda x: x.get("likes", 0), reverse=True)[:3]
        for i, model in enumerate(top_overall, 1):
            source_info = f" ({model.get('source', 'unknown')})" if "source" in model else ""
            print(f"{i}. {model['model_id']} ({model['task']}) - {model['likes']} likes{source_info}")

        print("\nTasks with most models available:")
        task_counts = [(task, len(models)) for task, models in data.items()]
        task_counts.sort(key=lambda x: x[1], reverse=True)
        for task, count in task_counts[:5]:
            print(f"  {task}: {count} models")

        # Show source distribution if available
        if all_models and "source" in all_models[0]:
            source_counts = defaultdict(int)
            for model in all_models:
                source_counts[model.get("source", "unknown")] += 1

            print("\nModel source distribution:")
            for source, count in source_counts.items():
                print(f"  {source}: {count} models")


class HuggingFaceModelScraper:
    def __init__(self, delay: float = 1.0):
        """
        Initialize the scraper with optional delay between requests.

        Args:
            delay: Delay in seconds between requests to be respectful
        """
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )
        self.delay = delay
        self.base_url = "https://huggingface.co"

    def extract_model_content(self, url: str) -> Dict:
        """
        Extract all relevant content from a Hugging Face model page.

        Args:
            url: The Hugging Face model page URL

        Returns:
            Dictionary containing extracted model information
        """
        try:
            # Add delay to be respectful
            time.sleep(self.delay)

            response = self.session.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            model_data = {
                "url": url,
                "model_name": self._extract_model_name(soup, url),
                "description": self._extract_description(soup),
                "readme_content": self._extract_readme(soup),
                "metadata": self._extract_metadata(soup),
                "tags": self._extract_tags(soup),
                "model_card": self._extract_model_card(soup),
                "files": self._extract_files_info(soup),
                "pipeline_tag": self._extract_pipeline_tag(soup),
                "library_name": self._extract_library_name(soup),
            }

            return model_data

        except requests.RequestException as e:
            print(f"Error fetching URL {url}: {e}")
            return {"error": str(e), "url": url}
        except Exception as e:
            print(f"Error processing content: {e}")
            return {"error": str(e), "url": url}

    def _extract_model_name(self, soup: BeautifulSoup, url: str) -> str:
        """Extract model name from the page."""
        # Try to get from title tag
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text().strip()
            # Remove " · Hugging Face" suffix if present
            if " · Hugging Face" in title:
                return title.replace(" · Hugging Face", "").strip()

        # Fallback: extract from URL
        return url.split("/")[-1] if url.split("/") else "Unknown"

    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Extract model description."""
        # Look for meta description
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            return meta_desc.get("content").strip()

        # Look for description in various possible locations
        desc_selectors = [
            'div[data-target="ModelHeader"] p',
            ".model-card-description",
            "div.text-gray-700",
            "p.text-gray-600",
        ]

        for selector in desc_selectors:
            desc_elem = soup.select_one(selector)
            if desc_elem:
                return desc_elem.get_text().strip()

        return ""

    def _extract_readme(self, soup: BeautifulSoup) -> str:
        """Extract README/model card content in original markdown format."""
        # First try to get raw markdown from API endpoint
        raw_markdown = self._get_raw_markdown_from_api(soup)
        if raw_markdown:
            return raw_markdown

        # Fallback: Try to extract from page source
        raw_markdown = self._extract_markdown_from_page(soup)
        if raw_markdown:
            return raw_markdown

        # Last resort: Convert HTML back to approximate markdown
        return self._convert_html_to_markdown(soup)

    def _get_raw_markdown_from_api(self, soup: BeautifulSoup) -> str:
        """Try to get raw markdown content from HuggingFace API."""
        try:
            # Extract model path from current URL
            current_url = soup.find("link", {"rel": "canonical"})
            if not current_url:
                return ""

            url = current_url.get("href", "")
            if not url:
                return ""

            # Parse model owner/name from URL
            parts = url.replace("https://huggingface.co/", "").split("/")
            if len(parts) < 2:
                return ""

            model_path = f"{parts[0]}/{parts[1]}"

            # Try to fetch raw README.md from the API
            api_url = f"https://huggingface.co/{model_path}/raw/main/README.md"

            time.sleep(self.delay)  # Be respectful
            response = self.session.get(api_url)

            if response.status_code == 200:
                return response.text

        except Exception as e:
            print(f"Could not fetch raw markdown from API: {e}")

        return ""

    def _extract_markdown_from_page(self, soup: BeautifulSoup) -> str:
        """Try to extract markdown from script tags or data attributes."""
        # Look for script tags that might contain markdown
        scripts = soup.find_all("script")
        for script in scripts:
            if script.string:
                # Look for markdown content in various formats
                if "README.md" in script.string or "# " in script.string:
                    # Try to extract markdown from JSON data
                    try:
                        import json

                        # Look for JSON that might contain markdown
                        json_matches = re.findall(r"\{.*?\}", script.string, re.DOTALL)
                        for match in json_matches:
                            try:
                                data = json.loads(match)
                                if isinstance(data, dict):
                                    # Look for markdown in various keys
                                    for key in ["content", "markdown", "readme", "text"]:
                                        if key in data and isinstance(data[key], str):
                                            if len(data[key]) > 100 and "#" in data[key]:
                                                return data[key]
                            except json.JSONDecodeError:
                                continue
                    except:
                        pass

        return ""

    def _convert_html_to_markdown(self, soup: BeautifulSoup) -> str:
        """Convert HTML content back to approximate markdown format."""
        # Look for the main content area
        readme_selectors = [
            'div[data-target="ModelHeader"] + div',
            ".prose",
            "div.markdown",
            "article",
            'div[class*="readme"]',
            "div.model-card",
        ]

        content_elem = None
        for selector in readme_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                break

        if not content_elem:
            return ""

        # Convert HTML elements to markdown
        markdown_content = []

        for element in content_elem.find_all(
            [
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
                "p",
                "ul",
                "ol",
                "li",
                "pre",
                "code",
                "blockquote",
                "a",
                "strong",
                "em",
            ]
        ):
            if element.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                level = int(element.name[1])
                markdown_content.append(f"{'#' * level} {element.get_text().strip()}\n")

            elif element.name == "p":
                text = element.get_text().strip()
                if text:
                    markdown_content.append(f"{text}\n")

            elif element.name == "pre":
                code_text = element.get_text()
                # Check if it's a code block
                if element.find("code"):
                    markdown_content.append(f"```\n{code_text}\n```\n")
                else:
                    markdown_content.append(f"```\n{code_text}\n```\n")

            elif element.name == "code" and element.parent.name != "pre":
                markdown_content.append(f"`{element.get_text()}`")

            elif element.name == "ul":
                for li in element.find_all("li", recursive=False):
                    markdown_content.append(f"- {li.get_text().strip()}\n")
                markdown_content.append("\n")

            elif element.name == "ol":
                for i, li in enumerate(element.find_all("li", recursive=False), 1):
                    markdown_content.append(f"{i}. {li.get_text().strip()}\n")
                markdown_content.append("\n")

            elif element.name == "blockquote":
                quote_text = element.get_text().strip()
                for line in quote_text.split("\n"):
                    if line.strip():
                        markdown_content.append(f"> {line.strip()}\n")
                markdown_content.append("\n")

            elif element.name == "a":
                href = element.get("href", "")
                text = element.get_text().strip()
                if href and text:
                    markdown_content.append(f"[{text}]({href})")

            elif element.name == "strong":
                markdown_content.append(f"**{element.get_text()}**")

            elif element.name == "em":
                markdown_content.append(f"*{element.get_text()}*")

        # Join and clean up the markdown
        markdown_text = "".join(markdown_content)

        # Clean up extra newlines
        markdown_text = re.sub(r"\n{3,}", "\n\n", markdown_text)

        return markdown_text.strip()

    def _extract_metadata(self, soup: BeautifulSoup) -> Dict:
        """Extract model metadata like downloads, likes, etc."""
        metadata = {}

        # Look for download count
        download_elem = soup.find(text=re.compile(r"\d+\s*downloads?"))
        if download_elem:
            downloads = re.search(r"([\d,]+)\s*downloads?", download_elem, re.I)
            if downloads:
                metadata["downloads"] = downloads.group(1).replace(",", "")

        # Look for likes
        like_elem = soup.find(text=re.compile(r"\d+\s*likes?"))
        if like_elem:
            likes = re.search(r"(\d+)\s*likes?", like_elem, re.I)
            if likes:
                metadata["likes"] = likes.group(1)

        # Look for model size
        size_elem = soup.find(text=re.compile(r"\d+\.?\d*\s*[KMGT]?B"))
        if size_elem:
            size = re.search(r"(\d+\.?\d*\s*[KMGT]?B)", size_elem)
            if size:
                metadata["model_size"] = size.group(1)

        return metadata

    def _extract_tags(self, soup: BeautifulSoup) -> List[str]:
        """Extract model tags."""
        tags = []

        # Look for tag elements
        tag_selectors = ["span.tag", ".badge", "[data-tag]", 'span[class*="tag"]']

        for selector in tag_selectors:
            tag_elems = soup.select(selector)
            for elem in tag_elems:
                tag_text = elem.get_text().strip()
                if tag_text and tag_text not in tags:
                    tags.append(tag_text)

        return tags

    def _extract_model_card(self, soup: BeautifulSoup) -> Dict:
        """Extract structured model card information."""
        model_card = {}

        # Look for JSON-LD structured data
        json_scripts = soup.find_all("script", type="application/ld+json")
        for script in json_scripts:
            try:
                data = json.loads(script.string)
                if isinstance(data, dict):
                    model_card.update(data)
            except json.JSONDecodeError:
                continue

        return model_card

    def _extract_files_info(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract information about model files."""
        files = []

        # Look for file listings
        file_elems = soup.select('a[href*="/blob/"], a[href*="/resolve/"]')
        for elem in file_elems:
            href = elem.get("href", "")
            filename = href.split("/")[-1] if href else ""
            if filename:
                files.append({"filename": filename, "url": urljoin(self.base_url, href)})

        return files

    def _extract_pipeline_tag(self, soup: BeautifulSoup) -> str:
        """Extract the pipeline tag/task type."""
        # Look for pipeline tag in various locations
        pipeline_selectors = ["[data-pipeline-tag]", 'span[class*="pipeline"]', 'div[class*="task"]']

        for selector in pipeline_selectors:
            elem = soup.select_one(selector)
            if elem:
                return elem.get("data-pipeline-tag") or elem.get_text().strip()

        return ""

    def _extract_library_name(self, soup: BeautifulSoup) -> str:
        """Extract the library name (e.g., transformers, sentence-transformers)."""
        # Look for library information
        lib_elem = soup.find(text=re.compile(r"(transformers|sentence-transformers|diffusers|timm)", re.I))
        if lib_elem:
            match = re.search(r"(transformers|sentence-transformers|diffusers|timm)", lib_elem, re.I)
            if match:
                return match.group(1).lower()

        return ""
