import logging
import shutil
import sys
import zipfile
from pathlib import Path

from .constants import WEBUI_INPUT_MARKER, WEBUI_INPUT_REQUEST

logger = logging.getLogger(__name__)


# We do not suggest using the agent with zip files.
# But MLEBench contains datasets zip files.
# https://github.com/WecoAI/aideml/blob/main/aide/utils/__init__.py
def clean_up_dataset(path: Path):
    for item in path.rglob("__MACOSX"):
        if item.is_dir():
            shutil.rmtree(item)
    for item in path.rglob(".DS_Store"):
        if item.is_file():
            item.unlink()


# We do not suggest using the agent with zip files.
# But MLEBench contains datasets zip files.
# https://github.com/WecoAI/aideml/blob/main/aide/utils/__init__.py
def extract_archives(path):
    """
    Unzips all .zip files within `path` and cleans up task dir

    Args:
        path: Path object or string path to directory containing zip files

    [TODO] handle nested zips
    """
    # Convert string path to Path object if necessary
    if isinstance(path, str):
        path = Path(path)

    for zip_f in path.rglob("*.zip"):
        f_out_dir = zip_f.with_suffix("")
        # special case: the intended output path already exists (maybe data has already been extracted by user)
        if f_out_dir.exists():
            logger.debug(f"Skipping {zip_f} as an item with the same name already exists.")
            # if it's a file, it's probably exactly the same as in the zip -> remove the zip
            # [TODO] maybe add an extra check to see if zip file content matches the colliding file
            if f_out_dir.is_file() and f_out_dir.suffix != "":
                zip_f.unlink()
                continue

        logger.debug(f"Extracting: {zip_f}")
        f_out_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_f, "r") as zip_ref:
            zip_ref.extractall(f_out_dir)

        # remove any unwanted files
        clean_up_dataset(f_out_dir)
        contents = list(f_out_dir.iterdir())

        # special case: the zip contains a single dir/file with the same name as the zip
        if len(contents) == 1 and contents[0].name == f_out_dir.name:
            sub_item = contents[0]
            # if it's a dir, move its contents to the parent and remove it
            if sub_item.is_dir():
                logger.debug(f"Special handling (child is dir) enabled for: {zip_f}")
                for f in sub_item.rglob("*"):
                    shutil.move(f, f_out_dir)
                sub_item.rmdir()
            # if it's a file, rename it to the parent and remove the parent
            elif sub_item.is_file():
                logger.debug(f"Special handling (child is file) enabled for: {zip_f}")
                sub_item_tmp = sub_item.rename(f_out_dir.with_suffix(".__tmp_rename"))
                f_out_dir.rmdir()
                sub_item_tmp.rename(f_out_dir)

        zip_f.unlink()


def get_user_input_webui(prompt: str) -> str:
    """Get user input in WebUI environment"""
    # Send special marker with the prompt
    print(f"{WEBUI_INPUT_REQUEST} {prompt}", flush=True)

    # Read from stdin - Flask will send the user input here
    while True:
        line = sys.stdin.readline().strip()
        if line.startswith(WEBUI_INPUT_MARKER):
            # Extract the actual user input after the marker
            user_input = line[len(WEBUI_INPUT_MARKER) :].strip()
            logger.debug(f"Received WebUI input: {user_input}")
            return user_input
