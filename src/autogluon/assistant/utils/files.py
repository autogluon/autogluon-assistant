import os

from ..constants import TEXT_EXTENSIONS


def is_text_file(filename):
    _, ext = os.path.splitext(filename)
    return ext.lower() in TEXT_EXTENSIONS
