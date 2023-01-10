import os


def make_dir_if_not_exists(path) -> None:
    """creates a directory if it does not exist already"""
    if not os.path.exists(path):
        os.makedirs(path)
