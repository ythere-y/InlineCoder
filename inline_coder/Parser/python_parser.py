import tree_sitter_python as tspy
from tree_sitter import Language, Parser, Node
from typing import Optional
from inline_coder.utils.logger_config import LoggerUtils


class PythonFileParser:

    PYTHON_LANGUAGE = Language(tspy.language())
    python_paser = Parser(Language(tspy.language()))

    @staticmethod
    def traverse(node: Node, type: str, res_list: Optional[list[Node]] = None):
        if res_list is None:
            res_list = []
        if node.type == "function_definition":
            res_list.append(node)
        for child in node.children:
            PythonFileParser.traverse(child, type, res_list)
        return res_list

    @classmethod
    def ExtractFunction(cls, file_path: str) -> list[Node]:
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
        tree = cls.python_paser.parse(bytes(code, "utf8"))
        root_node = tree.root_node
        function_nodes = PythonFileParser.traverse(root_node, "function_definition")
        return function_nodes


if __name__ == "__main__":
    logger = LoggerUtils.get_main_logger(
        name="PythonParser",
        log_file="LOGS/Parser/PythonParser.log",
    )
