import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from ..llm import ChatLLMFactory

logger = logging.getLogger(__name__)

def extract_section(text: str, section_marker: str) -> str:
    """
    Extract a section from the text starting with the section_marker and ending with a double newline.
    
    Args:
        text: The full text to extract from
        section_marker: The marker that indicates the start of the section
        
    Returns:
        The extracted section text or empty string if marker not found
    """
    if section_marker not in text:
        return ""
        
    start = text.find(section_marker)
    end = text.find("\n\n", start)
    
    # If there's no double newline, it's the last section in the text
    if end == -1:
        return text[start:].strip()+"\n\n"
    else:
        return text[start:end].strip()+"\n\n"

def generate_error_prompt(
    task_prompt: str,
    data_prompt: str,
    user_prompt: str,
    python_code: str,
    bash_script: str,
    tutorial_prompt: str,
    error_message: str,
    llm_config,
    output_folder: Optional[str],
    max_error_message_length: int = 2000,
    error_summary: bool = True,
    error_fix: bool = True,
) -> str:
    """Generate an error prompt by analyzing the error message and providing guidance for code improvement.
    Args:
        task_prompt: Description of the data science task
        data_prompt: Description of the data
        user_prompt: Instructions from the user
        python_code: Previous Python code that generated the error
        bash_script: Previous Bash script that generated the error
        tutorial_prompt: Tutorial information
        error_message: Error message from the last run
        llm_config: Configuration for the LLM
        output_folder: Optional folder to save the results
        max_error_message_length: Maximum length for error message
        error_summary: Whether to summarize the error using LLM
        error_fix: Whether to include fix suggestions in the prompt
    Returns:
        str: Formatted error prompt with analysis and suggestions
    """
    try:
        # Truncate error message if needed
        if len(error_message) > max_error_message_length:
            error_message = (
                error_message[: max_error_message_length // 2]
                + "\n...(truncated)\n"
                + error_message[-max_error_message_length // 2 :]
            )
        
        # If error_summary is False, just return the truncated error message
        if not error_summary:
            return f"Error Summary: {error_message}"
            
        # Create LLM instance
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        llm = ChatLLMFactory.get_chat_model(
            llm_config, session_name=f"error_summarizer_{timestamp}"
        )
        
        # Construct context for error analysis
        context = f"""{task_prompt}
{data_prompt}
{user_prompt}
Previous Python Code:
```python
{python_code}
```
Previous Bash Script to Execute the Python Code:
```bash
{bash_script}
```
{tutorial_prompt}
Error Message:
{error_message}"""

        # Modify analysis prompt based on error_fix flag
        if error_fix:
            analysis_prompt = """Analyze the error message and context provided. Your response MUST contain exactly two short paragraphs as follows:

ERROR SUMMARY: Provide a brief, technical description of the error in 1-3 sentences. Focus only on identifying the root cause and affected component without background explanations.

SUGGESTED FIX: Offer specific debugging directions in 1-3 sentences. Do not include actual code or commands, only tactical debugging guidance.

Each paragraph must be concise (maximum 3 sentences). Do not include general advice, explanations beyond the direct debugging strategy, or any additional paragraphs."""
        else:
            analysis_prompt = """Analyze the error message and context provided. Your response MUST contain exactly one short paragraphs as follows:

ERROR SUMMARY: Provide a brief, technical description of the error in 1-3 sentences. Focus only on identifying the root cause and affected component without background explanations.

Each paragraph must be concise (maximum 3 sentences). Do not include general advice, explanations beyond the direct debugging strategy, or any additional paragraphs."""

        context = context + "\n\n" + analysis_prompt
    
        # Get error analysis from LLM
        full_analysis = llm.assistant_chat(context)
        
        # Extract only the error summary and suggested fix paragraphs using the same logic
        error_analysis = ""
            
        # Extract ERROR SUMMARY
        error_summary_text = extract_section(full_analysis, "ERROR SUMMARY:")
        if error_summary_text:
            error_analysis += error_summary_text
        
        # Extract SUGGESTED FIX if error_fix is True
        if error_fix:
            suggested_fix_text = extract_section(full_analysis, "SUGGESTED FIX:")
            if suggested_fix_text:
                error_analysis += suggested_fix_text

        if not error_analysis:
            error_analysis = full_analysis
        
        # Save results if output folder is provided
        if output_folder:
            save_error_analysis(
                Path(output_folder), context, error_analysis, error_message
            )
            
        return error_analysis
    except Exception as e:
        logger.error(f"Error generating error prompt: {e}")
        # Fallback to basic error message if LLM analysis fails
        return f"Error Summary: {str(error_message)[:max_error_message_length]}"

def save_error_analysis(
    output_folder: Path, context: str, error_analysis: str, original_error: str
) -> None:
    """Save error analysis results to output folder."""
    try:
        output_folder.mkdir(parents=True, exist_ok=True)
        analysis_data = {
            "context": context,
            "error_analysis": error_analysis,
            "original_error": original_error,
            "timestamp": str(datetime.now()),
        }
        with open(output_folder / "error_analysis.json", "w", encoding="utf-8") as f:
            json.dump(analysis_data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving error analysis: {e}")

