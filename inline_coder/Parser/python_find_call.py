import logging
import tree_sitter_python as tspy
from tree_sitter import Language, Parser, Node


class PaserLoader:
    PYTHON_LANGUAGE = Language(tspy.language())
    python_paser = Parser(Language(tspy.language()))


class FindCallError(Exception):
    logger = logging.getLogger("FindCallError")

    def __init__(self, conti: bool = False, message="FindCall Error!"):
        self.message = message
        self.conti = conti
        super().__init__(self.message)
        self.logger.error(f"continue = {self.conti}, message = {self.message}")


class FindCall:
    @classmethod
    def find_call(cls, blob: str):
        tree = PaserLoader.python_paser.parse(bytes(blob, "utf8"))
        root_node = tree.root_node
        query = PaserLoader.PYTHON_LANGUAGE.query(
            """
(call
  function: (attribute) @func_full)

(call
  function: (identifier) @func_full)
"""
        )
        captures = query.captures(root_node)
        call_list = []
        if "func_full" in captures:
            for cap in captures["func_full"]:
                if isinstance(cap, Node):
                    func_name = cap.text.decode("utf8")  # type: ignore
                    if func_name not in call_list:
                        call_list.append(
                            {
                                "name": func_name,
                                "node": cap,
                            }
                        )
        return call_list
