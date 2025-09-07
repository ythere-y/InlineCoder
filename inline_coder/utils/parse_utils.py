from typing import List, Tuple, Dict, Any, Set


def get_docstring_summary(docstring: str) -> str:
    """Get the first lines of the documentation comment up to the empty lines."""
    if "\n\n" in docstring:
        return docstring.split("\n\n")[0]
    elif "@" in docstring:
        return docstring[
            : docstring.find("@")
        ]  # This usually is the start of a JavaDoc-style @param comment.
    return docstring


def strip_c_style_comment_delimiters(comment: str) -> str:
    comment_lines = comment.split("\n")
    cleaned_lines = []
    for l in comment_lines:
        l = l.strip()
        if l.endswith("*/"):
            l = l[:-2]
        if l.startswith("*"):
            l = l[1:]
        elif l.startswith("/**"):
            l = l[3:]
        elif l.startswith("//"):
            l = l[2:]
        cleaned_lines.append(l.strip())
    return "\n".join(cleaned_lines)


def traverse_type(node, results: List, kind: str, current_depth: int = 0) -> None:
    """
    collect a specific type of Node
    """
    if current_depth > 800:
        return
    if node == None:
        return
    if node.type == kind:
        results.append(node)
    if not node.children:
        return
    for n in node.children:
        traverse_type(n, results, kind, current_depth + 1)


def extract_range_from_span(start_point, end_point, blob: str) -> str:
    lines = blob.split("\n")
    line_start = start_point[0]
    line_end = end_point[0]
    char_start = start_point[1]
    char_end = end_point[1]

    if line_start != line_end:
        return "\n".join(
            [lines[line_start][char_start:]]
            + lines[line_start + 1 : line_end]
            + [lines[line_end][:char_end]]
        )
    else:
        return lines[line_start][char_start:char_end]
