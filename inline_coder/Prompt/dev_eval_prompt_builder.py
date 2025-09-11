from tree_sitter import Node
from pathlib import Path
from typing import Optional

from inline_coder.Prompt.constants import ExampleFunction, InfoInjection
from inline_coder.Preprocess.dev_eval_preprocess import DevEvalUtils
from inline_coder.Parser.python_parser import PythonFileParser


class DevEvalPromptBuilder:

    example_function = {
        "name": "process_user_data",
        "signature": "def process_user_data(users: list, min_age: int) -> list:",
        "body": "filtered_users = []\nfor user in users:\n    if not isinstance(user, dict):\n        continue\n    age = user.get('age', 0)\n    AgeValidation.validate(age);\n    name = user.get('name', '')\n    NameValidation.validate(name);\n    if age >= min_age and name:\n        user['is_adult'] = True\n        filtered_users.append(user)\n    else:\n        user['is_adult'] = False\nreturn filtered_users",
        "downstream": "AgeValidation.validate, NameValidation.validate",
    }

    @classmethod
    def __get_target_function_node(cls, sample) -> Node:
        target_file = Path(DevEvalUtils.SourceCodePath) / sample["completion_path"]
        function_nodes = PythonFileParser.ExtractFunction(target_file)
        target_node = None
        same_name_nodes = [
            func_node
            for func_node in function_nodes
            if func_node.child_by_field_name("name").text.decode()  # type: ignore
            == sample["namespace"].split(".")[-1]
        ]
        if len(same_name_nodes) == 0:
            pass
        elif len(same_name_nodes) > 1:
            same_line_nodes = [
                func_node
                for func_node in same_name_nodes
                if func_node.start_point[0] + 1 == sample["signature_position"][0]
            ]
            if len(same_line_nodes) == 0:
                closest_node = min(
                    same_name_nodes,
                    key=lambda x: abs(
                        x.start_point[0] + 1 - sample["signature_position"][0]
                    ),
                )
                target_node = closest_node
                sample["signature_position"] = (
                    closest_node.start_point[0] + 1,
                    [
                        child_node
                        for child_node in closest_node.children
                        if child_node.type == ":"
                    ][0].end_point[0]
                    + 1,
                )
            else:
                target_node = same_line_nodes[0]
        else:
            target_node = same_name_nodes[0]
            if target_node.start_point[0] + 1 != sample["signature_position"][0]:
                sample["signature_position"] = (
                    target_node.start_point[0] + 1,
                    [
                        child_node
                        for child_node in target_node.children
                        if child_node.type == ":"
                    ][0].end_point[0]
                    + 1,
                )
        assert target_node is not None, f"target node is None, sample: {sample}"
        return target_node

    @staticmethod
    def __get_argument_prompt(sample, indentation: str) -> str:
        argument_text = sample["requirement"]["Arguments"]
        argument_text = (
            argument_text[:-1].replace("\n", "\n" + indentation) + "\n"
            if argument_text.endswith("\n")
            else argument_text.replace("\n", "\n" + indentation)
        )
        argument_prompt = f"{indentation}{argument_text}" if argument_text else ""
        return argument_prompt

    @staticmethod
    def __get_functionality_prompt(sample, indentation: str) -> str:
        functionality_text = sample["requirement"]["Functionality"]
        functionality_text = (
            functionality_text[:-1].replace("\n", "\n" + indentation) + "\n"
            if functionality_text.endswith("\n")
            else functionality_text.replace("\n", "\n" + indentation)
        )
        functionality_prompt = (
            f"{indentation}{functionality_text}" if functionality_text else ""
        )
        return functionality_prompt

    @classmethod
    def BasePrompt(cls, sample) -> str:
        target_function_node = cls.__get_target_function_node(sample)
        target_function_name = sample["namespace"].split(".")[-1]
        if "target_file" not in sample:
            target_file = Path(DevEvalUtils.SourceCodePath) / sample["completion_path"]
            with open(target_file, "r") as f:
                sample["target_file"] = f.read()
        target_signature = "\n".join(
            sample["target_file"].split("\n")[
                sample["signature_position"][0] - 1 : sample["signature_position"][1]
            ]
        )
        sample["target_signature"] = target_signature
        body_indent = target_function_node.start_point[1] + 4
        indentation = " " * body_indent
        argument_prompt = cls.__get_argument_prompt(sample, indentation)
        functionality_prompt = cls.__get_functionality_prompt(sample, indentation)
        context_above = "context above the function"
        context_below = "context below the function"
        start_line = target_function_node.start_point[0]
        end_line = target_function_node.end_point[0] + 1
        context_above = "\n".join(sample["target_file"].split("\n")[:start_line]) + "\n"
        if context_above.strip() == "":
            context_above = ""
        context_below = "\n".join(sample["target_file"].split("\n")[end_line:])
        template = f"""Please complete the {target_function_name} function in the middle of a file.

The contexts above the function are:
```Python
{context_above}
```

The contexts below the function are:
```Python
{context_below}
```

The code to be completed is:
```Python
{target_signature}

{indentation}\"\"\"
{functionality_prompt}
{indentation}Input-Output Arguments
{argument_prompt}
{indentation}\"\"\"
```

The response should follow the format of the example below:  
{{
    "target_function_name": '''{ExampleFunction.example['name']}''',
    "target_function_signature": '''{ExampleFunction.example['signature']}''',
    "target_function_body": '''{ExampleFunction.example['body']}''',
}}
Please make sure to follow the format strictly. The response should be a valid JSON object.
Your response:
{{
    "target_function_name": '''''',
    "target_function_signature": '''''',
    "target_function_body": '''''',
}}"""
        return template

    @classmethod
    def DevEvalBasePromptWithDownStream(cls, sample) -> str:
        target_function_node = cls.__get_target_function_node(sample)
        target_function_name = sample["namespace"].split(".")[-1]
        if "target_file" not in sample:
            target_file = Path(DevEvalUtils.SourceCodePath) / sample["completion_path"]
            with open(target_file, "r") as f:
                sample["target_file"] = f.read()
        target_signature = "\n".join(
            sample["target_file"].split("\n")[
                sample["signature_position"][0] - 1 : sample["signature_position"][1]
            ]
        )
        sample["target_signature"] = target_signature
        body_indent = target_function_node.start_point[1] + 4
        indentation = " " * body_indent
        argument_prompt = cls.__get_argument_prompt(sample, indentation)
        functionality_prompt = cls.__get_functionality_prompt(sample, indentation)
        context_above = "context above the function"
        context_below = "context below the function"
        start_line = target_function_node.start_point[0]
        end_line = target_function_node.end_point[0] + 1
        context_above = "\n".join(sample["target_file"].split("\n")[:start_line]) + "\n"
        if context_above.strip() == "":
            context_above = ""
        context_below = "\n".join(sample["target_file"].split("\n")[end_line:])
        template = f"""Please complete the {target_function_name} function in the middle of a file. And also provide the downstream functions that are called in the target function.

The contexts above the function are:
```Python
{context_above}
```

The contexts below the function are:
```Python
{context_below}
```

The code to be completed is:
```Python
{target_signature}

{indentation}\"\"\"
{functionality_prompt}
{indentation}Input-Output Arguments
{argument_prompt}
{indentation}\"\"\"
```

The response should follow the format of the example below:  
{{
    "target_function_name": '''{ExampleFunction.example['name']}''',
    "target_function_signature": '''{ExampleFunction.example['signature']}''',
    "target_function_body": '''{ExampleFunction.example['body']}''',
    "down_stream": '''{ExampleFunction.example['downstream']}'''
}}
Please make sure to follow the format strictly. The response should be a valid JSON object.
Your response:
{{
    "target_function_name": '''''',
    "target_function_signature": '''''',
    "target_function_body": '''''',
    "down_stream":'''''',
}}"""
        return template

    @classmethod
    def UpDownInjectPrompt(
        cls,
        sample,
        inlined_result_list: list,
        useful_downstream: list,
        solution: str,
        ppl_result: Optional[dict] = None,
    ):
        target_function_node = cls.__get_target_function_node(sample)
        target_function_name = sample["namespace"].split(".")[-1]
        if "target_file" not in sample:
            target_file = Path(DevEvalUtils.SourceCodePath) / sample["completion_path"]
            with open(target_file, "r") as f:
                sample["target_file"] = f.read()
        target_signature = "\n".join(
            sample["target_file"].split("\n")[
                sample["signature_position"][0] - 1 : sample["signature_position"][1]
            ]
        )
        sample["target_signature"] = target_signature
        body_indent = target_function_node.start_point[1] + 4
        indentation = " " * body_indent
        argument_prompt = cls.__get_argument_prompt(sample, indentation)
        functionality_prompt = cls.__get_functionality_prompt(sample, indentation)
        context_above = "context above the function"
        context_below = "context below the function"
        start_line = target_function_node.start_point[0]
        end_line = target_function_node.end_point[0] + 1
        context_above = "\n".join(sample["target_file"].split("\n")[:start_line]) + "\n"
        if context_above.strip() == "":
            context_above = ""
        context_below = "\n".join(sample["target_file"].split("\n")[end_line:])
        PPL_explaination = InfoInjection.get_ppl_explaination(ppl_result)
        downstream_functions_prompt = InfoInjection.get_downstream_prompt(
            useful_downstream
        )
        upstream_functions_prompt = InfoInjection.get_upstream_prompt(
            sample, inlined_result_list
        )

        template = f"""Please complete the {target_function_name} function in the middle of a file.

The contexts above the function are:
```Python
{context_above}
```

The contexts below the function are:
```Python
{context_below}
```

{upstream_functions_prompt}

{downstream_functions_prompt}

Below are current version of the target function{PPL_explaination}. Try to keep the comments in this version ::
```Python
{solution}
```

The code to be completed is:
```Python
{target_signature}

{indentation}\"\"\"
{functionality_prompt}
{indentation}Input-Output Arguments
{argument_prompt}
{indentation}\"\"\"
```

The response should follow the format of the example below:  
{{
    "target_function_name": '''{ExampleFunction.example['name']}''',
    "target_function_signature": '''{ExampleFunction.example['signature']}''',
    "target_function_body": '''{ExampleFunction.example['body']}''',
}}
Please make sure to follow the format strictly. The response should be a valid JSON object.
Your response:
{{
    "target_function_name": '''''',
    "target_function_signature": '''''',
    "target_function_body": '''''',
}}"""
        return template
