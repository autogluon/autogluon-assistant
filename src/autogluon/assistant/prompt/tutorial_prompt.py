import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, NamedTuple, Optional

from ..llm import ChatLLMFactory
from ..tools_registry import get_tool_tutorials_folder

logger = logging.getLogger(__name__)


class TutorialInfo(NamedTuple):
    """Stores information about a tutorial"""

    path: Path
    title: str
    summary: str


def get_all_tutorials(tool_name: str, condensed: bool = False) -> List[TutorialInfo]:
    """Get all tutorial files of the tool, optionally returning condensed versions.

    Args:
        tool_name: Name of the ML tool to use in codes
        condensed: Whether to return condensed versions of tutorials

    Returns:
        List of TutorialInfo containing file path, title, and summary
    """
    tutorial_dir = get_tool_tutorials_folder(tool_name, condensed=condensed)

    tutorial_files = []
    for file_path in tutorial_dir.rglob("*.md"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().split("\n")

                # Find title (first line starting with #)
                title = next(
                    (
                        line.lstrip("#").strip()
                        for line in content
                        if line.strip().startswith("#")
                    ),
                    "",
                )

                # Find summary (line starting with "Summary: ")
                summary = next(
                    (
                        line.replace("Summary:", "").strip()
                        for line in content
                        if line.strip().startswith("Summary:")
                    ),
                    "",
                )

                if title:  # Only add if we found a title
                    tutorial_files.append(TutorialInfo(file_path, title, summary))

        except Exception as e:
            logger.warning(f"Error reading tutorial file {file_path}: {e}")
            continue

    return tutorial_files


def select_relevant_tutorials(
    tutorials: List[TutorialInfo],
    task_prompt: str,
    data_prompt: str,
    user_prompt: str,
    error_prompt: str,
    llm_config,
    max_num_tutorials: int,
    use_tutorial_summary: bool = True,
) -> List[TutorialInfo]:
    """Select most relevant tutorials using LLM scoring based on titles and summaries."""

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    llm_select_tutorial = ChatLLMFactory.get_chat_model(
        llm_config, session_name=f"tutorial_selector_{timestamp}"
    )

    context = f"""Task: {task_prompt}
Data: {data_prompt}
User Question: {user_prompt}
Previous Error: {error_prompt}"""

    tutorials_info = "\n".join(
        f"{i+1}. Title: {tutorial.title}\n   Summary: {tutorial.summary if use_tutorial_summary and tutorial.summary else '(No summary available)'}"
        for i, tutorial in enumerate(tutorials)
    )

    prompt = f"""Given the following context and list of tutorials with their summaries, select the {max_num_tutorials} most relevant tutorials for helping with this task. Consider how well each tutorial's title and summary match the task, data, user question, and any errors.

Context:
{context}

Available Tutorials:
{tutorials_info}

IMPORTANT: Respond ONLY with the numbers of the selected tutorials (up to {max_num_tutorials}) separated by commas. 
For example: "1,3,4" or "2,5" or just "1" if only one is relevant.
DO NOT include any other text, explanation, or formatting in your response."""

    try:
        content = llm_select_tutorial.assistant_chat(prompt)
        content = content.split("\n")[0]
        content = "".join(char for char in content if char.isdigit() or char == ",")

        if not content:
            logger.warning("No valid indices found in LLM response")
            return tutorials[:max_num_tutorials]

        try:
            selected_indices = [
                int(idx.strip()) - 1 for idx in content.split(",") if idx.strip()
            ]
        except ValueError as e:
            logger.warning(f"Error parsing indices from LLM response: {e}")
            return tutorials[:max_num_tutorials]

        selected_tutorials = []
        for idx in selected_indices:
            if 0 <= idx < len(tutorials):
                selected_tutorials.append(tutorials[idx])

        if len(selected_tutorials) > max_num_tutorials:
            selected_tutorials = selected_tutorials[:max_num_tutorials]
        return selected_tutorials

    except Exception as e:
        logger.warning(f"Error selecting tutorials: {e}")
        raise e


def format_tutorial_content(
    tutorial: TutorialInfo,
    max_length: int,
) -> str:
    """Format a single tutorial's content with truncation if needed."""
    try:
        with open(tutorial.path, "r", encoding="utf-8") as f:
            content = f.read()

        # Truncate if needed
        if len(content) > max_length:
            content = content[:max_length] + "\n...(truncated)"

        formatted = f"""### {tutorial.title}
        
        {content}
        """
        return formatted

    except Exception as e:
        logger.warning(f"Error formatting tutorial {tutorial.path}: {e}")
        return ""


def save_selection_results(
    output_folder: Path,
    selected_tutorials: List[TutorialInfo],
    formatted_tutorials: List[str],
    tutorial_prompt: str,
) -> None:
    """Save selection results and prompt to output folder."""
    try:
        output_folder.mkdir(parents=True, exist_ok=True)

        selection_data = [
            {
                "path": str(tutorial.path),
                "title": tutorial.title,
                "summary": tutorial.summary,
            }
            for tutorial in selected_tutorials
        ]

        with open(
            output_folder / "selected_tutorials.json", "w", encoding="utf-8"
        ) as f:
            json.dump(selection_data, f, indent=2)

        contents_folder = output_folder / "tutorial_contents"
        contents_folder.mkdir(exist_ok=True)

        for i, content in enumerate(formatted_tutorials, 1):
            with open(contents_folder / f"tutorial_{i}.md", "w", encoding="utf-8") as f:
                f.write(content)

        with open(output_folder / "tutorial_prompt.txt", "w", encoding="utf-8") as f:
            f.write(tutorial_prompt)

    except Exception as e:
        logger.error(f"Error saving selection results: {e}")


def generate_tutorial_prompt(
    task_prompt: str,
    data_prompt: str,
    user_prompt: str,
    error_prompt: str,
    tool_name: str,
    llm_config,
    output_folder: Optional[str],
    max_num_tutorials: int = 3,
    max_tutorial_length: int = 9999,
    condense_tutorials: bool = False,
    use_tutorial_summary: bool = True,
) -> str:
    """Generate a tutorial prompt by selecting relevant tutorials.

    Args:
        task_prompt: Describe the data science task
        data_prompt: Describe the data
        user_prompt: Instructions from the user
        error_prompt: Error from last run
        tool_name: Name of the ML tool to use in codes
        llm_config: Configuration for the LLM
        output_folder: Optional folder to save results
        max_num_tutorials: Maximum number of tutorials to include
        max_tutorial_length: Maximum length for each tutorial
        condense_tutorials: Whether to use condensed versions of tutorials

    Returns:
        str: Formatted tutorial prompt containing selected tutorials
    """
    tutorials = get_all_tutorials(tool_name, condensed=condense_tutorials)
    if not tutorials:
        logger.warning(f"No tutorials found for {tool_name}")
        return ""

    selected_tutorials = select_relevant_tutorials(
        tutorials,
        task_prompt,
        data_prompt,
        user_prompt,
        error_prompt,
        llm_config,
        max_num_tutorials,
        use_tutorial_summary,
    )

    if not selected_tutorials:
        return ""

    formatted_tutorials = []
    for tutorial in selected_tutorials:
        formatted = format_tutorial_content(
            tutorial,
            max_tutorial_length,
        )
        if formatted:
            formatted_tutorials.append(formatted)

    if not formatted_tutorials:
        return ""

    prompt = "RELEVANT TUTORIALS:\n" + "\n\n".join(formatted_tutorials)

    if output_folder:
        output_path = Path(output_folder)
        save_selection_results(
            output_path, selected_tutorials, formatted_tutorials, prompt
        )

    return prompt
