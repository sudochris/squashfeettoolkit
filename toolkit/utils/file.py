import os
from os import path

def join_folders(lst):
    assert len(lst) > 0, "At least one element must be provided!"

    folder = lst[0]
    for p in lst[1:]:
        folder = path.join(folder, p)

    return folder

def exists(file_path):
    return path.exists(file_path)

def make_dirs(folder_path):
    return os.makedirs(folder_path, exist_ok=True)