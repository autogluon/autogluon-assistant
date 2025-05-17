from typing import List


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
