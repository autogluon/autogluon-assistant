import logging
from typing import List

from ..tools_registry import TutorialInfo, get_tool_tutorials_folder
from .base_prompt import BasePrompt

logger = logging.getLogger(__name__)


# TODO: move this util function to tools_registry
def get_all_tutorials(selected_tool: str, condensed: bool = False) -> List[TutorialInfo]:
    """Get all tutorial files of the tool, optionally returning condensed versions.

    Args:
        selected_tool: Name of the ML tool to use in codes
        condensed: Whether to return condensed versions of tutorials

    Returns:
        List of TutorialInfo containing file path, title, and summary
    """
    tutorial_dir = get_tool_tutorials_folder(selected_tool, condensed=condensed)

    tutorial_files = []
    for file_path in tutorial_dir.rglob("*.md"):  # TODO: support other file formats
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().split("\n")

                # Find title (first line starting with #)
                title = next(
                    (line.lstrip("#").strip() for line in content if line.strip().startswith("#")),
                    "",
                )

                # Find summary (line starting with "Summary: ")
                summary = next(
                    (line.replace("Summary:", "").strip() for line in content if line.strip().startswith("Summary:")),
                    "",
                )

                if title:  # Only add if we found a title
                    tutorial_files.append(TutorialInfo(file_path, title, summary))

        except Exception as e:
            logger.warning(f"Error reading tutorial file {file_path}: {e}")
            continue

    return tutorial_files


class RerankerPrompt(BasePrompt):
    """Handles prompts for tutorial retrieval and selection"""

    def default_template(self) -> str:
        """Default template for tutorial selection"""
        return """
Given the following context and list of tutorials with their summaries, select the {max_num_tutorials} most relevant tutorials for helping with this task. Consider how well each tutorial's title and summary match the task, data, user question, and any errors.

### Task Description
{task_description}

### Data Structures
{data_prompt}

### User Instruction
{user_input}

### Previous Error Analysis
{error_analysis}

Available Tutorials:
{tutorials_info}

IMPORTANT: Respond ONLY with the numbers of the selected tutorials (up to {max_num_tutorials}) separated by commas. 
For example: "1,3,4" or "2,5" or just "1" if only one is relevant.
DO NOT include any other text, explanation, or formatting in your response.
"""

    def build(self) -> str:
        """Build a prompt for the LLM to select relevant tutorials."""

        # Get tutorial information
        selected_tool = self.manager.selected_tool
        use_tutorial_summary = self.manager.config.use_tutorial_summary

        # Get retrieved tutorials from manager
        self.tutorials = self.manager.tutorial_retrieval

        if not self.tutorials:
            logger.warning(f"No tutorials found for {selected_tool}")
            return ""

        # Format tutorials info for selection
        tutorials_info = "\n".join(
            f"{i+1}. Title: {tutorial.title}\n   Summary: {tutorial.summary if use_tutorial_summary and tutorial.summary else '(No summary available)'}"
            for i, tutorial in enumerate(self.tutorials)
        )

        # Format the prompt using the template
        prompt = self.template.format(
            task_description=self.manager.task_description,
            data_prompt=self.manager.data_prompt,
            user_input=self.manager.user_input,
            error_analysis=self.manager.all_previous_error_prompts,
            tutorials_info=tutorials_info,
            max_num_tutorials=self.manager.config.max_num_tutorials,
        )

        self.manager.save_and_log_states(
            content=prompt, save_name="reranker_prompt.txt", per_iteration=True, add_uuid=False
        )

        return prompt

    def parse(self, response: str) -> List[int]:
        """Parse the LLM response to extract selected tutorial indices."""

        self.manager.save_and_log_states(
            content=response, save_name="reranker_response.txt", per_iteration=True, add_uuid=False
        )

        try:
            # Clean the response - take first line and keep only digits and commas
            content = response.split("\n")[0]
            content = "".join(char for char in content if char.isdigit() or char == ",")

            if not content:
                logger.warning("No valid indices found in LLM response")
                return []

            # Parse comma-separated indices
            selected_indices = []
            try:
                indices = [int(idx.strip()) - 1 for idx in content.split(",") if idx.strip()]
                selected_indices = [idx for idx in indices if idx >= 0]  # Filter out negative indices
            except ValueError as e:
                logger.warning(f"Error parsing indices from LLM response: {e}")
                return []

            selected_tutorials = []
            for idx in selected_indices:
                if 0 <= idx < len(self.tutorials):
                    selected_tutorials.append(self.tutorials[idx])

            if len(selected_tutorials) > self.manager.config.max_num_tutorials:
                selected_tutorials = selected_tutorials[: self.manager.config.max_num_tutorials]

            self.manager.save_and_log_states(
                content=selected_tutorials, save_name="selected_tutorials.txt", per_iteration=True, add_uuid=False
            )
            return selected_tutorials

        except Exception as e:
            logger.warning(f"Error parsing tutorial selection response: {e}")
            return []
