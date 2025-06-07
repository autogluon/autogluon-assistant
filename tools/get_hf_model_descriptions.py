#!/usr/bin/env python3
"""
Hugging Face Model Page Content Extractor

This script extracts content from Hugging Face model pages including:
- Model name and description
- README content
- Model metadata (downloads, likes, etc.)
- Model card information
- Tags and task information
"""

import json
import re
import time
from typing import Dict, List
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


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


def main():
    """Example usage of the scraper."""
    scraper = HuggingFaceModelScraper(delay=1.0)

    # Example URLs
    test_urls = [
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
        "https://huggingface.co/bert-base-uncased",
        "https://huggingface.co/gpt2",
    ]

    for url in test_urls:
        print(f"\n{'='*60}")
        print(f"Extracting content from: {url}")
        print("=" * 60)

        model_data = scraper.extract_model_content(url)

        if "error" in model_data:
            print(f"Error: {model_data['error']}")
            continue

        # Print extracted information
        print(f"Model Name: {model_data['model_name']}")
        print(f"Description: {model_data['description'][:200]}...")
        print(f"Pipeline Tag: {model_data['pipeline_tag']}")
        print(f"Library: {model_data['library_name']}")
        print(f"Tags: {', '.join(model_data['tags'][:5])}")  # First 5 tags
        print(f"Metadata: {model_data['metadata']}")
        print(f"Files: {len(model_data['files'])} files found")

        if model_data["readme_content"]:
            print(f"README (first 300 chars): {model_data['readme_content'][:300]}...")

        # Save to JSON file
        safe_model_name = model_data["model_name"].replace("/", "_").replace("\\", "_")
        output_filename = f"{safe_model_name}_data.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(model_data, f, indent=2, ensure_ascii=False)
        print(f"Data saved to: {output_filename}")

        # Save README to separate .md file
        if model_data["readme_content"]:
            readme_filename = f"{safe_model_name}_README.md"
            with open(readme_filename, "w", encoding="utf-8") as f:
                f.write(model_data["readme_content"])
            print(f"README saved to: {readme_filename}")
        else:
            print("No README content found to save")


if __name__ == "__main__":
    main()
