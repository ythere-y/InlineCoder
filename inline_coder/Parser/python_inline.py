import rich
import logging
from typing import Dict, Any, Optional
from rich.columns import Columns
from rich.panel import Panel
import tree_sitter_python as tspy
from tree_sitter import Language, Parser, Node
from inline_coder.Parser.parser_utils import ParserUtils
from inline_coder.utils.data_utils import string_diff


class PaserLoader:
    PYTHON_LANGUAGE = Language(tspy.language())
    python_paser = Parser(Language(tspy.language()))


class InlineError(Exception):
    logger = logging.getLogger("InlineError")

    def __init__(self, conti: bool = False, message="Inline Error!"):
        self.message = message
        self.conti = conti
        super().__init__(self.message)
        self.logger.error(f"continue = {self.conti}, message = {self.message}")


class PythonParser:
    logger = logging.getLogger("PythonParser")

    @classmethod
    def InlineMethod(
        cls, call_edge_info: Dict, display: bool = False
    ) -> Dict[str, Any]:
        if display:
            rich.print(
                Columns(
                    [
                        Panel(
                            call_edge_info["caller"],
                            title="caller_code",
                            border_style="blue",
                        ),
                        Panel(
                            call_edge_info["callee"],
                            title="callee_code",
                            border_style="green",
                        ),
                    ]
                )
            )
        CR_tree = PaserLoader.python_paser.parse(
            bytes(call_edge_info["caller"], "utf-8")
        )
        CE_tree = PaserLoader.python_paser.parse(
            bytes(call_edge_info["callee"], "utf-8")
        )

        def get_function_name(root_node: Node):
            query = PaserLoader.PYTHON_LANGUAGE.query(
                """
                (function_definition name: (identifier) @name)
                """
            )
            captures = query.captures(root_node)
            if "name" not in captures:
                raise InlineError(conti=True, message="function name not found!")
            if len(captures["name"]) == 0:
                raise InlineError(conti=True, message="function name not found!")
            if len(captures["name"]) > 1:
                cls.logger.error("function name found too much!, use the first found")
            if type(captures["name"][0].text) != bytes:
                raise InlineError(
                    conti=True, message="function name type not supported!"
                )
            return captures["name"][0].text.decode()

        def get_method_invocation(node):
            query = PaserLoader.PYTHON_LANGUAGE.query(
                """
                (call) @function_call
                """
            )
            captures = query.captures(node)
            if "function_call" not in captures:
                raise InlineError(conti=True, message="function call not found!")
            return captures["function_call"]

        def get_calling_arguments(
            head_node: Node, callee_name: str
        ) -> tuple[list[dict[str, Optional[str]]], Optional[Node]]:
            calling_list = get_method_invocation(head_node)
            target_call: Optional[Node] = None
            argument_list = []
            for _candidate_call in calling_list:

                _candidate_name = ParserUtils.get_text(
                    _candidate_call.child_by_field_name("function")
                )
                if "." in _candidate_name:
                    _candidate_name = _candidate_name.split(".")[-1]
                if _candidate_name == callee_name:
                    target_call = _candidate_call
            if target_call == None:
                raise InlineError(conti=True, message="Target call not found!")
            argument_list_node = target_call.child_by_field_name("arguments")
            if argument_list_node == None:
                return argument_list, target_call
            for _arg in argument_list_node.named_children:
                if _arg.type in [
                    "string",
                    "subscript",
                    "identifier",
                    "attribute",
                    "call",
                    "binary_operator",
                    "lambda",
                    "conditional_expression",
                ]:
                    argument_list.append(
                        {
                            "name": None,
                            "value": ParserUtils.get_text(_arg),
                        }
                    )
                elif _arg.type == "keyword_argument":
                    argument_list.append(
                        {
                            "name": ParserUtils.get_text(
                                _arg.child_by_field_name("name")
                            ),
                            "value": ParserUtils.get_text(
                                _arg.child_by_field_name("value")
                            ),
                        }
                    )
                elif _arg.type in ["dictionary_splat", "list_splat"]:
                    argument_list.append(
                        {
                            "name": None,
                            "value": ParserUtils.get_text(_arg.named_children[0]),
                        }
                    )
                else:
                    raise InlineError(
                        conti=True, message="argument type not supported!"
                    )
            return argument_list, target_call

        def get_function_parameters(head_node: Node):
            query = PaserLoader.PYTHON_LANGUAGE.query(
                """
                (function_definition
                    parameters: (parameters)@parameter_list
                )
                """
            )
            captures = query.captures(head_node)
            result = []
            if captures["parameter_list"] is None:
                raise InlineError(conti=True, message="function parameters not found!")
            parameters_node: Node = captures["parameter_list"][0]

            for _param in parameters_node.named_children:
                if _param.type == "identifier":
                    result.append({"parameter_name": ParserUtils.get_text(_param)})
                    continue
                else:
                    cur_res = {}
                    sub_identifer = [
                        __node
                        for __node in _param.children
                        if __node.type == "identifier"
                    ]
                    if len(sub_identifer) == 0:
                        raise InlineError(
                            conti=True, message="function parameters parser error!"
                        )
                    cur_res["parameter_name"] = ParserUtils.get_text(sub_identifer[0])
                    # .text.decode()
                    if "default" in _param.type:
                        cur_res["parameter_type"] = "with_default"
                        cur_res["default_value"] = ParserUtils.get_text(
                            _param.child_by_field_name("value")
                        )
                    result.append(cur_res)
            return result

        def get_identifiers(head_node: Node):
            query = PaserLoader.PYTHON_LANGUAGE.query(
                """
                (identifier) @identifier
                """
            )
            captures = query.captures(head_node)
            return captures["identifier"]

        def get_return_stm(head_node: Node):
            query = PaserLoader.PYTHON_LANGUAGE.query(
                """
                (return_statement) @ret_stm
                """
            )
            captures = query.captures(head_node)
            if "ret_stm" not in captures:
                return []
            return captures["ret_stm"]

        def callee_transform(
            head_node: Node, arguments_list: list[dict[str, Optional[str]]], blob: str
        ) -> str:
            new_blob = blob
            parameters_list = get_function_parameters(head_node)

            # build up a map between the parameters and the arguments
            parameters_mapping = {}
            for _param in parameters_list:
                if "default_value" in _param:
                    parameters_mapping[_param["parameter_name"]] = _param[
                        "default_value"
                    ]
            for _arg in arguments_list:
                if _arg["name"] != None:
                    parameters_mapping[_arg["name"]] = _arg["value"]
            no_name_arg_list = [
                _arg["value"] for _arg in arguments_list if _arg["name"] == None
            ]
            para_len = len(no_name_arg_list)
            for idx, _arg in enumerate(arguments_list):
                if idx >= para_len:
                    break
                if idx >= len(parameters_list):
                    raise InlineError(conti=True, message="argument number too much!")
                parameters_mapping[parameters_list[idx]["parameter_name"]] = _arg[
                    "value"
                ]

            # traverse and store replace position
            replace_queue = []
            identifiers_list = get_identifiers(head_node)
            for _identifier in identifiers_list:
                if _identifier.text == None:
                    continue
                _name = str(_identifier.text, "utf-8")
                if _name in parameters_mapping:
                    replace_queue.append(
                        {
                            "position": _identifier.byte_range,
                            "place_taker": parameters_mapping[_name],
                        }
                    )

            # replace return statement
            return_statements = get_return_stm(head_node)
            for ret_stm in reversed(return_statements):
                replace_queue.append(
                    {
                        "position": ret_stm.children[0].byte_range,
                        "place_taker": "result = ",
                    }
                )

            # sort replace queue
            replace_queue.sort(key=lambda x: x["position"][0], reverse=True)

            # replace all words from botton to top
            for _replace in replace_queue:
                new_blob = ParserUtils.span_replacement(
                    new_blob, _replace["position"], _replace["place_taker"]
                )

            return new_blob

        def get_body_block_info(head_node: Node):
            query = PaserLoader.PYTHON_LANGUAGE.query(
                """
                (function_definition
                body:(block) @block
                )
                """
            )
            captures = query.captures(head_node)
            if "block" not in captures:
                raise InlineError(conti=True, message="function body not found!")
            if len(captures["block"]) > 1:
                cls.logger.error("function body found too much!, use the first found")
            block_node = captures["block"][0]
            body_str = ParserUtils.get_text(block_node)
            body_indent = 4
            return body_indent, body_str

        def caller_transform(target_call: Node, blob: str) -> Dict[str, Any]:
            blob_lines = blob.split("\n")

            new_blob = blob

            result_info = {
                "new_blob": new_blob,
                "insert_position": None,
                "indent": None,
            }
            belonging_node = target_call.parent
            if belonging_node == None:
                raise InlineError(
                    conti=True, message="belonging node not found in caller!"
                )
            insert_position = target_call.start_point[0]
            indent = 0
            if belonging_node.type == "expression_statement":
                blob_lines = (
                    blob_lines[: target_call.start_point[0]]
                    + blob_lines[target_call.end_point[0] + 1 :]
                )
                new_blob = "\n".join(blob_lines)
                indent = belonging_node.start_point[1]
            else:
                new_blob = ParserUtils.span_replacement(
                    new_blob, target_call.byte_range, "result"
                )
                while "statement" not in belonging_node.type:
                    belonging_node = belonging_node.parent
                    if belonging_node == None:
                        raise InlineError(
                            conti=True,
                            message="belonging node not found in caller!",
                        )
                indent = belonging_node.start_point[1]
            result_info["new_blob"] = new_blob
            result_info["insert_position"] = insert_position
            result_info["indent"] = indent
            return result_info

        def insert_callee_body_into_caller(
            caller_insert_info: Dict[str, Any],
            insert_body_indent: int,
            insert_body_str: str,
        ) -> dict[str, Any]:
            insert_body_lines = insert_body_str.split("\n")
            gap = caller_insert_info["indent"] - insert_body_indent
            for idx, line in enumerate(insert_body_lines):
                if idx == 0:
                    line = " " * caller_insert_info["indent"] + line
                else:
                    if gap > 0:
                        line = " " * gap + line
                insert_body_lines[idx] = line

            insert_body_str = "\n".join(insert_body_lines)
            caller_body = caller_insert_info["new_blob"]
            caller_body_lines = caller_body.split("\n")
            insert_line_number = caller_insert_info["insert_position"]
            caller_body_lines.insert(insert_line_number, insert_body_str)
            caller_body = "\n".join(caller_body_lines)
            edit_lines = range(
                insert_line_number, insert_line_number + len(insert_body_lines)
            )
            edit_codes = insert_body_str
            insert_result = {
                "result": caller_body,
                "edit_lines": edit_lines,
                "edit_codes": edit_codes,
            }
            return insert_result

        inline_result = {
            "result": None,
            "edit_lines": [],
            "edit_codes": [],
        }
        try:
            callee_name = get_function_name(CE_tree.root_node)
            argument_list, calling_node = get_calling_arguments(
                CR_tree.root_node, callee_name
            )
            if calling_node == None:
                raise InlineError(conti=True, message="calling node not found!")
            new_CE_blob = callee_transform(
                CE_tree.root_node, argument_list, call_edge_info["callee"]
            )
            if display:
                string_diff(
                    str_org=call_edge_info["callee"],
                    str_new=new_CE_blob,
                    title="callee code after transform",
                    logger=cls.logger,
                )
            try:
                new_CE_tree = PaserLoader.python_paser.parse(
                    bytes(new_CE_blob, "utf-8")
                )
                CE_body_indent, CE_body_str = get_body_block_info(new_CE_tree.root_node)
            except InlineError as e:
                if e.conti:
                    new_CE_tree = CE_tree
                    CE_body_indent, CE_body_str = get_body_block_info(
                        new_CE_tree.root_node
                    )
                else:
                    raise e
            caller_insert_info = caller_transform(
                calling_node, call_edge_info["caller"]
            )
            inline_result = insert_callee_body_into_caller(
                caller_insert_info, CE_body_indent, CE_body_str
            )
            if display:
                string_diff(
                    str_org=call_edge_info["caller"],
                    str_new=inline_result["result"],
                    title="all after inline!~~",
                    logger=None,
                )
                rich.print(
                    f'[bold cyan] edit lines:{inline_result["edit_lines"]}[/bold cyan]'
                )
                rich.print(
                    f'[bold cyan] edit codes:{inline_result["edit_codes"]}[/bold cyan]'
                )
            return inline_result
        except InlineError as e:
            if e.conti:
                cls.logger.error(f"continue = {e.conti}, message = {e.message}")
                return {
                    "result": call_edge_info["caller"],
                    "edit_lines": [],
                    "edit_codes": [],
                }
            else:
                raise e
        except Exception as e:
            cls.logger.error(
                f"caller = {call_edge_info['caller']}, callee = {call_edge_info['callee']}"
            )
            raise e
