from typing import Union, Optional

from tree_sitter import Node


class ParserUtils:
    @staticmethod
    def span_replacement(
        blob: Union[str, bytes], byte_range, replace_str: Union[str, bytes]
    ) -> str:
        if isinstance(blob, bytes):
            bytes_blob = blob
        else:
            bytes_blob = blob.encode()

        # 这里检查replace_str是否已经是bytes类型
        if isinstance(replace_str, bytes):
            bytes_replace_str = replace_str
        else:
            bytes_replace_str = replace_str.encode()
        start, end = byte_range
        new_blob = bytes_blob[:start] + bytes_replace_str + bytes_blob[end:]
        return new_blob.decode()

    @staticmethod
    def get_text(node: Optional[Node]) -> str:
        if node is None:
            raise ValueError("node is None")
        if node.text is None:
            raise ValueError("node.text is None")
        return node.text.decode("utf-8")
