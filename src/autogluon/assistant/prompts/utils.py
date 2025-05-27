import logging
import re

logger = logging.getLogger(__name__)


def _extract_python_script(response):
    # Look for Python code blocks in the response
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()
    else:
        return None


def _extract_bash_script(response):
    # Look for Bash code blocks in the response
    pattern = r"```bash\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()
    else:
        return None


def extract_code(response, language):
    result = None

    if language == "python":
        result = _extract_python_script(response)
    elif language == "bash":
        result = _extract_bash_script(response)
    else:
        raise ValueError(f"Unsupported language: {language}")

    # If language-specific extraction failed, fallback to generic code blocks
    if result is None:
        logger.warning(f"No code block found for {language}, looking for the code wrapped without language specified")
        pattern = r"```\s*\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            result = matches[0].strip()

    # If still nothing found, return the full response
    if result is None:
        logger.warning(f"No code block found, return the full response instead: {response}")
        result = response

    return result
