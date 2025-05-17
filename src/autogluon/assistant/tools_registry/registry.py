import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..llm import ChatLLMFactory  # Import here to avoid circular imports
from .utils import split_markdown_into_chunks

logger = logging.getLogger(__name__)


class ToolsRegistry:
    def __init__(self):
        self.registry_path = Path(__file__).parent
        self.catalog_path = self.registry_path / "_common" / "catalog.json"
        self._tools_cache: Optional[Dict] = None

    @property
    def tools(self) -> Dict:
        """
        Lazy loading of tools information from catalog and individual tool.json files.

        Returns:
            Dict: Dictionary containing comprehensive tool information
        """
        if self._tools_cache is None:
            self._load_tools()
        return self._tools_cache

    def _load_tools(self) -> None:
        """
        Load and cache tools information from catalog.json and individual tool.json files.
        """
        try:
            with open(self.catalog_path, "r") as f:
                catalog = json.load(f)
        except Exception as e:
            logger.error(f"Error loading catalog.json: {e}")
            raise

        tools_info = {}
        for tool_name, tool_data in catalog["tools"].items():
            tool_path = self.registry_path / tool_data["path"] / "tool.json"
            tool_info = {
                "name": tool_name,
                "path": tool_data["path"],
                "version": tool_data["version"],
                "description": tool_data["description"],
            }

            try:
                with open(tool_path, "r") as f:
                    tool_json = json.load(f)
                    tool_info.update(
                        {
                            "features": tool_json.get("features", []),
                            "requirements": tool_json.get("requirements", []),
                            "prompt_template": tool_json.get("prompt_template", []),
                        }
                    )
            except Exception as e:
                logger.warning(f"Error loading tool.json for {tool_name}: {e}")
                tool_info.update(
                    {
                        "features": [],
                        "requirements": [],
                        "prompt_template": [],
                    }
                )

            tools_info[tool_name] = tool_info

        self._tools_cache = tools_info

    def register_tool(
        self,
        name: str,
        version: str,
        description: str,
        features: List[str] = None,
        requirements: List[str] = None,
        prompt_template: List[str] = None,
        tutorials_path: Optional[Path] = None,
        condense: bool = True,
        llm_config=None,
        max_length: int = 9999,
    ) -> None:
        """
        Register a new ML tool in the registry.

        Args:
            name: Name of the tool
            version: Version of the tool
            description: Description of the tool
            features: List of tool features
            requirements: List of tool requirements
            prompt_template: List of prompt template strings
            tutorials_path: Optional path to tutorials directory to copy
        """
        # Create tool directory
        tool_path = self.registry_path / name
        tool_path.mkdir(exist_ok=True)

        # Update catalog.json
        try:
            with open(self.catalog_path, "r") as f:
                catalog = json.load(f)
        except Exception as e:
            logger.error(f"Error reading catalog.json: {e}")
            raise

        catalog["tools"][name] = {
            "path": str(name),
            "version": version,
            "description": description,
        }

        with open(self.catalog_path, "w") as f:
            json.dump(catalog, f, indent=2)

        # Create tool.json
        tool_json = {
            "name": name,
            "version": version,
            "description": description,
            "features": features or [],
            "requirements": requirements or [],
            "prompt_template": prompt_template or [],
        }

        with open(tool_path / "tool.json", "w") as f:
            json.dump(tool_json, f, indent=2)

        # Handle tutorials if provided
        if tutorials_path and tutorials_path.exists():
            # Clear cache to force reload
            self._tools_cache = None
            self.add_tool_tutorials(
                tool_name=name,
                tutorials_source=tutorials_path,
                condense=condense,
                llm_config=llm_config,
                max_length=max_length,
            )

        # Clear cache to force reload
        self._tools_cache = None

    def add_tool_tutorials(
        self,
        tool_name: str,
        tutorials_source: Union[Path, str],
        condense: bool = True,
        llm_config=None,
        max_length: int = 9999,
        chunk_size: int = 8192,  # Size of chunks for processing
    ) -> None:
        """
        Add tutorials to a registered tool, with option to condense them using LLM.
        Processes tutorials chunk by chunk and maintains one LLM session per tutorial.
        Only generates summaries for condensed tutorials.

        Args:
            tool_name: Name of the tool
            tutorials_source: Path to source tutorials directory
            condense: Whether to create condensed versions and summaries
            llm_config: Configuration for the LLM (required if condense=True)
            max_length: Maximum length for condensed tutorials
            chunk_size: Size of chunks for processing tutorials
        """
        tool_path = self.get_tool_path(tool_name)
        if not tool_path:
            raise ValueError(f"Tool {tool_name} not found in registry")

        tutorials_source = Path(tutorials_source)
        if not tutorials_source.exists():
            raise FileNotFoundError(f"Tutorials source path {tutorials_source} not found")

        if condense and not llm_config:
            raise ValueError("llm_config is required when condense=True")

        # Create tutorials directory structure
        tutorials_dir = tool_path / "tutorials"
        tutorials_dir.mkdir(exist_ok=True)

        # Process each tutorial file
        for tutorial_file in tutorials_source.rglob("*.md"):
            relative_path = tutorial_file.relative_to(tutorials_source)
            destination = tutorials_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)

            # Read original content
            with open(tutorial_file, "r", encoding="utf-8") as f:
                content = f.read()
                first_line = content.split("\n")[0]
                title = first_line.lstrip("#").strip()

            # Create LLM instance for this tutorial with multi_turn enabled
            tutorial_config = llm_config.copy()
            tutorial_config.multi_turn = True  # Always enable multi-turn for tutorial processing
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            tutorial_id = f"{tool_name}_{relative_path.stem}_{timestamp}"
            llm = ChatLLMFactory.get_chat_model(tutorial_config, session_name=tutorial_id)

            if len(content) > 2 * chunk_size:
                # Process tutorial in chunks using smart markdown splitting
                chunks = split_markdown_into_chunks(content, max_chunk_size=chunk_size)
            else:
                chunks = [content]
            condensed_chunks = []

            for i, chunk in enumerate(chunks):
                context = "This is a continuation of the previous chunk. " if i > 0 else ""
                chunk_prompt = f"""{context}Condense this portion of the tutorial while preserving essential implementation details, code samples, and key concepts. Focus on:

1. Implementation details and techniques
2. Code snippets with necessary context
3. Critical configurations and parameters
4. Important warnings and best practices

Chunk {i+1}/{len(chunks)}:
{chunk}

Provide the condensed content in markdown format."""

                condensed_chunk = llm.assistant_chat(chunk_prompt)
                condensed_chunks.append(condensed_chunk)

            # Combine chunks and generate summary
            condensed_content = "\n\n".join(condensed_chunks)

            # Generate summary using the same LLM instance
            summary_prompt = f"""Generate a concise summary (within 100 words) of this tutorial that helps a code generation LLM understand:
1. What specific implementation knowledge or techniques it can find in this tutorial
2. What coding tasks this tutorial can help with
3. Key features or functionalities covered

Tutorial content:
{condensed_content}

Provide the summary in a single paragraph starting with "Summary: "."""

            tutorial_summary = llm.assistant_chat(summary_prompt)
            if not tutorial_summary.startswith("Summary: "):
                tutorial_summary = "Summary: " + tutorial_summary

            # Truncate if needed while preserving complete sections
            if len(condensed_content) > max_length:
                last_section = condensed_content[:max_length].rfind("\n#")
                if last_section > 0:
                    truncate_point = last_section
                else:
                    truncate_point = condensed_content[:max_length].rfind("\n\n")
                    if truncate_point == -1:
                        truncate_point = max_length

                condensed_content = condensed_content[:truncate_point] + "\n\n...(truncated)"

            # Write original content with summary
            with open(destination, "w", encoding="utf-8") as f:
                f.write(f"{tutorial_summary}\n\n")
                f.write(content)

            # Write condensed version with summary
            condensed_dir = tool_path / "condensed_tutorials"
            condensed_dir.mkdir(exist_ok=True)
            condensed_path = condensed_dir / relative_path
            condensed_path.parent.mkdir(parents=True, exist_ok=True)

            with open(condensed_path, "w", encoding="utf-8") as f:
                f.write(f"# Condensed: {title}\n\n")
                f.write(f"{tutorial_summary}\n\n")
                f.write(
                    "*This is a condensed version that preserves essential implementation details and context.*\n\n"
                )
                f.write(condensed_content)

    def unregister_tool(self, tool_name: str) -> None:
        """
        Remove a tool from the registry.

        Args:
            tool_name: Name of the tool to remove
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found in registry")

        # Remove from catalog.json
        with open(self.catalog_path, "r") as f:
            catalog = json.load(f)

        catalog["tools"].pop(tool_name, None)

        with open(self.catalog_path, "w") as f:
            json.dump(catalog, f, indent=2)

        # Remove tool directory
        tool_path = self.registry_path / tool_name
        if tool_path.exists():
            shutil.rmtree(tool_path)

        # Clear cache to force reload
        self._tools_cache = None

    def update_tool(
        self,
        tool_name: str,
        version: Optional[str] = None,
        description: Optional[str] = None,
        features: Optional[List[str]] = None,
        requirements: Optional[List[str]] = None,
        prompt_template: Optional[List[str]] = None,
    ) -> None:
        """
        Update an existing tool's information.

        Args:
            tool_name: Name of the tool to update
            version: New version (optional)
            description: New description (optional)
            features: New features list (optional)
            requirements: New requirements list (optional)
            prompt_template: New prompt template list (optional)
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found in registry")

        # Update catalog.json if needed
        if version or description:
            with open(self.catalog_path, "r") as f:
                catalog = json.load(f)

            if version:
                catalog["tools"][tool_name]["version"] = version
            if description:
                catalog["tools"][tool_name]["description"] = description

            with open(self.catalog_path, "w") as f:
                json.dump(catalog, f, indent=2)

        # Update tool.json if needed
        tool_path = self.registry_path / tool_name / "tool.json"
        with open(tool_path, "r") as f:
            tool_json = json.load(f)

        if version:
            tool_json["version"] = version
        if description:
            tool_json["description"] = description
        if features is not None:
            tool_json["features"] = features
        if requirements is not None:
            tool_json["requirements"] = requirements
        if prompt_template is not None:
            tool_json["prompt_template"] = prompt_template

        with open(tool_path, "w") as f:
            json.dump(tool_json, f, indent=2)

        # Clear cache to force reload
        self._tools_cache = None

    # Existing methods remain unchanged
    def get_tool(self, tool_name: str) -> Optional[Dict]:
        return self.tools.get(tool_name)

    def list_tools(self) -> List[str]:
        return list(self.tools.keys())

    def get_tool_path(self, tool_name: str) -> Optional[Path]:
        tool_info = self.get_tool(tool_name)
        if tool_info:
            return self.registry_path / tool_info["path"]
        return None

    def get_tool_version(self, tool_name: str) -> Optional[str]:
        tool_info = self.get_tool(tool_name)
        if tool_info:
            return tool_info["version"]
        return None

    def get_tool_prompt_template(self, tool_name: str) -> Optional[List[str]]:
        tool_info = self.get_tool(tool_name)
        if tool_info:
            return tool_info.get("prompt_template")
        return None

    def get_tools_by_feature(self, feature: str) -> List[str]:
        return [
            tool_name
            for tool_name, tool_info in self.tools.items()
            if feature.lower() in [f.lower() for f in tool_info.get("features", [])]
        ]

    def get_tool_tutorials_folder(self, tool_name: str, condensed: bool) -> Path:
        tool_path = self.get_tool_path(tool_name)
        if not tool_path:
            raise FileNotFoundError(f"Tool '{tool_name}' not found")

        if condensed:
            tutorials_folder = tool_path / "condensed_tutorials"
        else:
            tutorials_folder = tool_path / "tutorials"
        if not tutorials_folder.exists():
            raise FileNotFoundError(f"No tutorials directory found for tool '{tool_name}' at {tutorials_folder}")

        return tutorials_folder
