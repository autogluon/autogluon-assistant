import os
from pathlib import Path

import nbformat
from nbconvert import MarkdownExporter


def clear_notebook_outputs(notebook):
    """
    Clear all outputs from notebook cells.
    Args:
        notebook: nbformat.NotebookNode object
    """
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None


def contains_image(cell):
    """
    Check if a cell contains an image.
    Args:
        cell: notebook cell object
    Returns:
        bool: True if cell contains an image, False otherwise
    """
    if cell.cell_type == "markdown":
        # Check for markdown image syntax
        if "![" in cell.source:
            return True
    elif cell.cell_type == "code":
        # Check for image outputs
        for output in cell.outputs:
            if "data" in output and any(
                key.startswith("image/") for key in output.get("data", {})
            ):
                return True
    return False


def filter_notebook_content(notebook):
    """
    Filter out cells containing images from the notebook.
    Args:
        notebook: nbformat.NotebookNode object
    Returns:
        nbformat.NotebookNode: Filtered notebook
    """
    filtered_cells = []
    for cell in notebook.cells:
        if not contains_image(cell):
            filtered_cells.append(cell)
    notebook.cells = filtered_cells
    return notebook


def convert_notebook_to_markdown(notebook_path, output_path):
    """
    Convert a single Jupyter notebook to Markdown format, removing images and outputs.
    Args:
        notebook_path (str): Path to the input notebook
        output_path (str): Path where the markdown file should be saved
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Read the notebook
    with open(notebook_path, "r", encoding="utf-8") as notebook_file:
        notebook = nbformat.read(notebook_file, as_version=4)

    # Clear outputs and remove images
    clear_notebook_outputs(notebook)
    notebook = filter_notebook_content(notebook)

    # Save cleared notebook back to original file
    with open(notebook_path, "w", encoding="utf-8") as notebook_file:
        nbformat.write(notebook, notebook_file)

    # Convert to markdown
    markdown_exporter = MarkdownExporter()
    markdown, _ = markdown_exporter.from_notebook_node(notebook)

    # Write the markdown file
    with open(output_path, "w", encoding="utf-8") as md_file:
        md_file.write(markdown)


def batch_convert_notebooks(input_dir, output_dir):
    """
    Convert all Jupyter notebooks in a directory (and its subdirectories) to Markdown.
    Args:
        input_dir (str): Root directory containing the notebooks
        output_dir (str): Directory where markdown files will be saved
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all notebook files
    notebook_files = input_path.rglob("*.ipynb")
    for notebook_path in notebook_files:
        # Skip checkpoint files
        if ".ipynb_checkpoints" in str(notebook_path):
            continue

        # Create equivalent markdown path in output directory
        relative_path = notebook_path.relative_to(input_path)
        markdown_path = output_path / relative_path.with_suffix(".md")

        print(f"Converting: {notebook_path} -> {markdown_path}")
        try:
            convert_notebook_to_markdown(str(notebook_path), str(markdown_path))
        except Exception as e:
            print(f"Error converting {notebook_path}: {str(e)}")


if __name__ == "__main__":
    # Replace with your directory paths
    input_directory = "/media/agent/FlagEmbedding/Tutorials"
    output_directory = "/media/agent/AutoMLAgent/raw_tutorials/FlagEmbeddingTutorials"
    batch_convert_notebooks(input_directory, output_directory)
