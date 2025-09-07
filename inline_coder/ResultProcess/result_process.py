import json
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

from inline_coder.Parser.python_find_call import FindCall


class ResultProcess:
    PY_LANGUAGE = Language(tspython.language())
    parser = Parser(PY_LANGUAGE)

    @classmethod
    def clean_multi_line_comment(cls, code: str) -> str:
        result_code = ""
        result_code_lines = []
        code_lines = code.split("\n")
        if len(code_lines) <= 2:
            return code

        if '"""' in code_lines[1] or "'''" in code_lines[1]:
            quote_type = '"""' if '"""' in code_lines[1] else "'''"
            comment_num = 1
            in_comment = False
            for line in code_lines:
                if quote_type in line:
                    if in_comment:
                        in_comment = False
                    elif comment_num:
                        in_comment = True
                        comment_num -= 1
                    continue
                if not in_comment:
                    result_code_lines.append(line)
        result_code = "\n".join(result_code_lines)
        return result_code

    @classmethod
    def clean_and_get_function(cls, target_signature: str, gen_prediction: str) -> str:
        if "```json" in gen_prediction:
            # 从 ```json```中提取字符串内容
            gen_prediction = gen_prediction.split("```json")[1].strip()
            gen_prediction = gen_prediction.split("```")[0].strip()
        try:
            gen_json = json.loads(gen_prediction)
            function_body = gen_json.get("target_function_body", "    pass")
            result_function = f"{target_signature}\n{function_body}"
        except json.JSONDecodeError:
            result_function = f"{target_signature}\n    pass"
        return result_function

    @classmethod
    def with_downstream_parse(
        cls, target_signature: str, gen_prediction: str
    ) -> dict[str, str]:
        if "```json" in gen_prediction:
            # 从 ```json```中提取字符串内容
            gen_prediction = gen_prediction.split("```json")[1].strip()
            gen_prediction = gen_prediction.split("```")[0].strip()
        try:
            gen_json = json.loads(gen_prediction)
            function_body = gen_json.get("target_function_body", "    pass")
            down_stream = gen_json.get("down_stream", "")
            result_function = f"{target_signature}\n{function_body}\n{down_stream}"
            result = {
                "target_function": result_function,
                "target_signature": target_signature,
                "target_function_body": function_body,
                "down_stream": down_stream,
            }
        except json.JSONDecodeError:
            result_function = f"{target_signature}\n    pass"
            result = {
                "target_function": result_function,
                "target_signature": target_signature,
                "target_function_body": "    pass",
                "down_stream": "",
            }
        return result

    @classmethod
    def common_parse(cls, target_signature: str, gen_prediction: str) -> dict[str, str]:
        if "```json" in gen_prediction:
            # 从 ```json```中提取字符串内容
            gen_prediction = gen_prediction.split("```json")[1].strip()
            gen_prediction = gen_prediction.split("```")[0].strip()
        try:
            gen_json = json.loads(gen_prediction)
            function_body = gen_json.get("target_function_body", "    pass")
            result_function = f"{target_signature}\n{function_body}"
            result = {
                "target_function": result_function,
                "target_signature": target_signature,
                "target_function_body": function_body,
            }
        except json.JSONDecodeError:
            result_function = f"{target_signature}\n    pass"
            result = {
                "target_function": result_function,
                "target_signature": target_signature,
                "target_function_body": "    pass",
            }
        return result


class DownStreamAttach:
    @classmethod
    def GetUsefuleDownStreams(
        cls,
        target_function_name: str,
        function: str,
        ai_pred_downstream: str,
        candidate_functions: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        call_name_list = FindCall.find_call(function)
        short_name_list = []
        for name in call_name_list:
            short_name = name["name"].split(".")[-1]
            short_name_list.append(short_name)
        ai_suggested_downstream = ai_pred_downstream
        if ai_suggested_downstream != "":
            for name in ai_suggested_downstream.split(","):
                short_name = name.strip().split(".")[-1]
                short_name_list.append(short_name)
        short_name_list = list(set(short_name_list))
        # 从extracted_data中，根据data['project_path']和目前sample['project_path']进行匹配
        candidate_functions = candidate_functions
        useful_downstream = []
        for function_info in candidate_functions:
            for short_name in short_name_list:
                if short_name in function_info["name"]:
                    if target_function_name != function_info["name"]:
                        useful_downstream.append(function_info)

        unique_downstream = {}
        for item in useful_downstream:
            content = item["content"]
            if content not in unique_downstream:
                unique_downstream[content] = item
        useful_downstream = list(unique_downstream.values())
        return useful_downstream
