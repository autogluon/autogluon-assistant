from typing import Optional


def generate_user_prompt(
    user_input: Optional[str] = None, max_user_input_length: int = 9999
) -> str:
    """Generate a formatted user prompt from user input.

    Args:
        user_inputs: User input string to include in the prompt.
                    If None, returns an empty string.
        max_user_input_length: Maximum allowed length for user input.

    Returns:
        str: Formatted user prompt with wrapped and truncated input.
    """
    if not user_input:
        return ""

    # Truncate if needed
    if len(user_input) > max_user_input_length:
        user_input = user_input[: max_user_input_length - 3] + "..."

    # Create the prompt with section header
    prompt = f"USER INPUTS:\n{user_input.strip()}"

    return prompt
