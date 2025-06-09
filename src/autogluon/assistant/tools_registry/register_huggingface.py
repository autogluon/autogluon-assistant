#!/usr/bin/env python3
"""
Hugging Face Tool Registration Script

This script registers Hugging Face as an ML library tool in the registry by:
1. Fetching top models across all tasks
2. Extracting detailed model descriptions and documentation
3. Creating organized documentation files
4. Registering the tool with the registry
"""

import logging
from pathlib import Path
from typing import Dict, List

from omegaconf import OmegaConf

from .registry import ToolsRegistry
from .utils import HuggingFaceModelScraper, HuggingFaceModelsFetcher

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class HuggingFaceToolRegistrar:
    def __init__(self, output_dir: str = "hf_tutorials", top_models_per_task: int = 3):
        """
        Initialize the Hugging Face tool registrar.

        Args:
            output_dir: Directory to save tutorial files
            top_models_per_task: Number of top models to fetch per task
        """
        self.output_dir = Path(output_dir)
        self.top_models_per_task = top_models_per_task
        self.models_fetcher = HuggingFaceModelsFetcher()
        self.model_scraper = HuggingFaceModelScraper(delay=0.5)  # Reduced delay for faster processing
        self.registry = ToolsRegistry()

        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

    def fetch_top_models(self) -> Dict[str, List[Dict]]:
        """
        Fetch top models for all tasks from Hugging Face.

        Returns:
            Dictionary with task names as keys and model lists as values
        """
        logger.info(f"Fetching top {self.top_models_per_task} models for each task...")

        # Get top liked models
        top_liked = self.models_fetcher.get_top_models_all_tasks(n=self.top_models_per_task, sort_by="likes")

        # Get top downloaded models
        top_downloaded = self.models_fetcher.get_top_models_all_tasks(n=self.top_models_per_task, sort_by="downloads")

        # Merge the results
        merged_models = self.models_fetcher.merge_top_models(
            top_liked, top_downloaded, max_per_task=self.top_models_per_task * 2
        )

        logger.info(f"Successfully fetched models for {len(merged_models)} tasks")
        return merged_models

    def create_model_documentation(self, models_by_task: Dict[str, List[Dict]]) -> None:
        """
        Create detailed documentation files for models organized by task.

        Args:
            models_by_task: Dictionary of models organized by task
        """
        logger.info("Creating model documentation files...")

        # Create individual model tutorial files directly in output directory
        for task, models in models_by_task.items():
            logger.info(f"Processing {len(models)} models for task: {task}")

            for model in models:
                self.create_model_tutorial(model, task)

        # Create master index file
        self.create_master_index(models_by_task)

    def create_model_tutorial(self, model: Dict, task: str) -> None:
        """
        Create a detailed tutorial file for a specific model using scraped content.

        Args:
            model: Model information dictionary
            task: Task name for this model
        """
        model_id = model["model_id"]
        safe_model_name = self.sanitize_filename(model_id.replace("/", "_"))

        # Create filename: [task] model_identifier.md
        filename = f"[{task}] {safe_model_name}.md"
        tutorial_file = self.output_dir / filename

        logger.info(f"Creating tutorial for {model_id}...")

        # Try to get detailed model information using scraper
        detailed_info = None
        if model.get("url"):
            try:
                logger.info(f"Fetching detailed info for {model_id}...")
                detailed_info = self.model_scraper.extract_model_content(model["url"])
            except Exception as e:
                logger.warning(f"Could not fetch detailed info for {model_id}: {e}")

        # Create comprehensive model tutorial content
        content = f"""# {model_id} - {task.replace('-', ' ').title()}

## Model Overview

**Model ID**: `{model_id}`  
**Task**: {task}  
**URL**: {model.get('url', 'N/A')}  
**Likes**: {model.get('likes', 'N/A'):,}  
**Downloads**: {model.get('downloads', 'N/A'):,}  
**Library**: {model.get('library_name', 'N/A')}  
**Pipeline Tag**: {model.get('pipeline_tag', 'N/A')}  
**Source Ranking**: {model.get('source', 'N/A')}  

"""

        if detailed_info and not detailed_info.get("error"):
            # Add description
            if detailed_info.get("description"):
                content += f"""## Description

{detailed_info['description']}

"""

            # Add tags
            if detailed_info.get("tags"):
                content += f"""## Tags

`{' | '.join(detailed_info['tags'][:15])}`

"""

            # Add the full README content if available
            if detailed_info.get("readme_content"):
                content += f"""## Model Documentation

{detailed_info['readme_content']}

"""

            # Add metadata if available
            if detailed_info.get("metadata"):
                content += """## Additional Metadata

"""
                for key, value in detailed_info["metadata"].items():
                    content += f"- **{key.title()}**: {value}\n"
                content += "\n"
        else:
            # Fallback content if scraping failed
            content += f"""## Description

This is a {task.replace('-', ' ')} model from Hugging Face. Detailed documentation may be available at the model's Hugging Face page.

"""

        # Write the tutorial file
        with open(tutorial_file, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Created tutorial: {filename}")

    def create_master_index(self, models_by_task: Dict[str, List[Dict]]) -> None:
        """
        Create a master index file for all tasks and models.

        Args:
            models_by_task: Dictionary of models organized by task
        """
        content = """# Hugging Face Models Documentation

This documentation provides comprehensive tutorials for top Hugging Face models across different tasks.

## Available Models by Task

"""

        # Group models by task and create links
        for task, models in sorted(models_by_task.items()):
            content += f"### {task.replace('-', ' ').title()} ({len(models)} models)\n\n"

            for model in models:
                safe_model_name = self.sanitize_filename(model["model_id"].replace("/", "_"))
                filename = f"[{task}] {safe_model_name}.md"
                content += f"- [{model['model_id']}](./{filename})\n"
            content += "\n"

        content += """

## Quick Navigation by Category

### Natural Language Processing
"""

        nlp_tasks = [
            task
            for task in models_by_task.keys()
            if any(
                nlp_term in task.lower()
                for nlp_term in [
                    "text",
                    "language",
                    "translation",
                    "question",
                    "summarization",
                    "classification",
                    "generation",
                    "fill-mask",
                    "token-classification",
                ]
            )
        ]
        for task in sorted(nlp_tasks):
            models = models_by_task[task]
            content += f"- **{task}**: {len(models)} models\n"

        content += """
### Computer Vision
"""

        cv_tasks = [
            task
            for task in models_by_task.keys()
            if any(
                cv_term in task.lower()
                for cv_term in [
                    "image",
                    "vision",
                    "detection",
                    "segmentation",
                    "depth",
                    "keypoint",
                    "object",
                    "video",
                    "unconditional",
                ]
            )
        ]
        for task in sorted(cv_tasks):
            models = models_by_task[task]
            content += f"- **{task}**: {len(models)} models\n"

        content += """
### Audio Processing
"""

        audio_tasks = [
            task
            for task in models_by_task.keys()
            if any(
                audio_term in task.lower()
                for audio_term in ["audio", "speech", "voice", "automatic-speech-recognition"]
            )
        ]
        for task in sorted(audio_tasks):
            models = models_by_task[task]
            content += f"- **{task}**: {len(models)} models\n"

        content += """
### Multimodal
"""

        multimodal_tasks = [
            task
            for task in models_by_task.keys()
            if any(
                mm_term in task.lower()
                for mm_term in [
                    "any-to-any",
                    "visual-question",
                    "document-question",
                    "image-text",
                    "video-text",
                    "audio-text",
                ]
            )
        ]
        for task in sorted(multimodal_tasks):
            models = models_by_task[task]
            content += f"- **{task}**: {len(models)} models\n"

        content += f"""

## Statistics

- **Total Tasks**: {len(models_by_task)}
- **Total Models**: {sum(len(models) for models in models_by_task.values())}
- **Tutorial Files**: {sum(len(models) for models in models_by_task.values())}

## How to Use These Tutorials

Each tutorial file contains:

1. **Model Overview**: Basic information and statistics
2. **Description**: Detailed model description from Hugging Face
3. **Full Documentation**: Complete README content from the model page
4. **Quick Start**: Ready-to-use code examples
5. **Integration Tips**: Best practices and optimization suggestions

## Common Usage Patterns

### For Text Tasks
```python
from transformers import pipeline

# Quick pipeline approach
pipe = pipeline("task-name", model="model-id")
result = pipe("Your input text")
```

### For Vision Tasks
```python
from transformers import pipeline

# Image processing pipeline
pipe = pipeline("image-classification", model="model-id")
result = pipe("path/to/image.jpg")
```

### For Audio Tasks
```python
from transformers import pipeline

# Audio processing pipeline
pipe = pipeline("automatic-speech-recognition", model="model-id")
result = pipe("path/to/audio.wav")
```

## Installation Requirements

```bash
pip install transformers torch torchvision torchaudio
```

For specific models, additional dependencies might be required. Check individual tutorial files for model-specific requirements.

---

*This documentation was automatically generated from the top Hugging Face models across all tasks.*
"""

        index_file = self.output_dir / "README.md"
        with open(index_file, "w", encoding="utf-8") as f:
            f.write(content)

    def register_tool(self) -> None:
        """
        Register Hugging Face as a tool in the registry.
        """
        logger.info("Registering Hugging Face tool...")

        # Define tool information
        tool_name = "huggingface"
        version = "1.0.0"
        description = "Here we collect top liked/downloaded models from huggingface for each task."

        features = ["All tasks supported in huggingface are available."]

        requirements = []

        prompt_template = []

        mlzero_dir = Path(__file__).parent.parent

        # Always load default config first
        default_config_path = mlzero_dir / "configs" / "default.yaml"
        if not default_config_path.exists():
            raise FileNotFoundError(f"Default config file not found: {default_config_path}")

        config = OmegaConf.load(default_config_path)

        # Register the tool
        self.registry.register_tool(
            name=tool_name,
            version=version,
            description=description,
            features=features,
            requirements=requirements,
            prompt_template=prompt_template,
            tutorials_path=self.output_dir,
            condense=True,
            max_length=16384,  # Reasonable length for condensed tutorials
            llm_config=config.llm,  # TODO: add customizable config
        )

        logger.info(f"Successfully registered {tool_name} tool with {len(features)} features")

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename for cross-platform compatibility.

        Args:
            filename: Original filename

        Returns:
            Sanitized filename
        """
        # Replace problematic characters
        filename = filename.replace("/", "_").replace("\\", "_")
        filename = filename.replace(":", "_").replace("*", "_")
        filename = filename.replace("?", "_").replace('"', "_")
        filename = filename.replace("<", "_").replace(">", "_")
        filename = filename.replace("|", "_").replace(" ", "_")

        # Remove consecutive underscores
        while "__" in filename:
            filename = filename.replace("__", "_")

        return filename.strip("_")

    def run_registration_process(self) -> None:
        """
        Execute the complete registration process.
        """
        try:
            logger.info("Starting Hugging Face tool registration process...")

            # Step 1: Fetch top models
            models_by_task = self.fetch_top_models()

            # Step 2: Create documentation
            self.create_model_documentation(models_by_task)

            # Step 3: Register the tool
            self.register_tool()

            logger.info("Hugging Face tool registration completed successfully!")
            logger.info(f"Documentation created in: {self.output_dir.absolute()}")

            # Print summary
            total_models = sum(len(models) for models in models_by_task.values())
            logger.info("Summary:")
            logger.info(f"  - Tasks processed: {len(models_by_task)}")
            logger.info(f"  - Models documented: {total_models}")
            logger.info(
                f"  - Documentation files created: {sum(len(models) + 1 for models in models_by_task.values()) + 1}"
            )

        except Exception as e:
            logger.error(f"Registration process failed: {e}")
            raise
