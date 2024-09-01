import os

from typing import Callable
from utilbox.parse_utils import str2path


def search_file_in_subfolder(curr_query: str, tgt_match_fn: Callable = None,
                             return_name: bool = False, return_sorted: bool = True):
    """
    Search for files in a directory and its subdirectories that satisfy a certain condition.

    Args:
        curr_query (str):
            Path of the directory to search in.
        tgt_match_fn (callable, optional):
            A function that takes a file name and returns a boolean value.
            If provided, only files that satisfy this condition are returned.
            If not provided, all files will be returned. Defaults to None.
        return_name (bool, optional):
            If True, return file names instead of file paths. Defaults to False.
        return_sorted (bool, optional):
            If True, the output list will be sorted in lexicographical order. Defaults to True.

    Returns:
        list: A list of file paths (return_name=False) or file names (return_name=True) that satisfy tgt_match_fn.
    """
    candidates = []
    curr_query = str2path(curr_query)

    # If the input query is a file path
    if os.path.isfile(curr_query):
        dir_name, node_name = os.path.dirname(curr_query), os.path.basename(curr_query)
        if tgt_match_fn is None or tgt_match_fn(node_name):
            # Skip the symbolic link and only return existing files
            if not os.path.islink(curr_query):
                return candidates + [curr_query] if not return_name else candidates + [node_name]
        else:
            raise RuntimeError(f"The file at the path {curr_query} does not satisfy the provided condition!")

    # If the input query is a directory path
    for node_name in os.listdir(curr_query):
        node_path = os.path.join(curr_query, node_name)
        if os.path.isdir(node_path):
            # Recursively search subdirectories
            candidates = candidates + search_file_in_subfolder(
                node_path, tgt_match_fn, return_name=return_name, return_sorted=return_sorted
            )
        elif tgt_match_fn is None or tgt_match_fn(node_name):
            # Skip the symbolic link and only return existing files
            if not os.path.islink(node_path):
                candidates = candidates + [node_path] if not return_name else candidates + [node_name]

    return sorted(candidates) if return_sorted else candidates
