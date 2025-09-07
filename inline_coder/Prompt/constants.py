from typing import Optional


class ExampleFunction:
    example = {
        "name": "process_user_data",
        "signature": "def process_user_data(users: list, min_age: int) -> list:",
        "body": "filtered_users = []\nfor user in users:\n    if not isinstance(user, dict):\n        continue\n    age = user.get('age', 0)\n    AgeValidation.validate(age);\n    name = user.get('name', '')\n    NameValidation.validate(name);\n    if age >= min_age and name:\n        user['is_adult'] = True\n        filtered_users.append(user)\n    else:\n        user['is_adult'] = False\nreturn filtered_users",
        "downstream": "AgeValidation.validate, NameValidation.validate",
    }


class InfoInjection:
    @classmethod
    def get_ppl_explaination(cls, ppl_result: Optional[dict] = None) -> str:
        PPL_explaination = ""
        if ppl_result is not None:
            ppl_score = ppl_result.get("ppl", 0)
            if ppl_score < 1.3:
                PPL_explaination = f". Current implementation is confident and the comments are good, Please refer to it and keep these comments."
            elif ppl_score < 2.0:
                PPL_explaination = f". Current implementation is somewhat uncertain and comments are reasonable. Please refer to it partially."
            else:
                PPL_explaination = ". Current implementation is not confident. Please consider regenerating it."
        return PPL_explaination

    @classmethod
    def get_upstream_prompt(cls, sample: dict, inlined_result_list: list):
        upstream_info_prompt = ""
        if inlined_result_list:
            upstream_info_prompt = "Below are examples of the functions calling the target functions and the result of inling the current target function into calling functions. Please make sure your implementation fits well while inlining into the caller function.:"
            for idx, inline_res in enumerate(inlined_result_list):
                caller_context = sample["upstream"][idx]["content"]
                inlined_code = inline_res["result"]
                if inline_res["edit_lines"] == []:
                    cur_str = f"""Caller function [{idx}]:
    ```Python
    {caller_context}
    ```
    """
                else:
                    cur_str = f"""Caller function [{idx}]:
    ```Python
    {caller_context}
    ```

    Below is the inlined result:
    {inlined_code}
    """
                upstream_info_prompt += cur_str
        return upstream_info_prompt

    @classmethod
    def get_downstream_prompt(cls, useful_downstream: list):
        downstream_functions_prompt = ""
        if useful_downstream:
            downstream_functions_prompt = (
                "\nBelow are the useful downstream functions:\n"
            )
            for idx, down_stream in enumerate(useful_downstream):
                cur_str = f"""Downstream function [{idx}]:
```Python
{down_stream['content']}
```
"""
                downstream_functions_prompt += cur_str
        return downstream_functions_prompt

    @classmethod
    def get_repo_coder_prompt(cls, repo_coder_function_list: list) -> str:
        if not repo_coder_function_list:
            return ""
        repo_coder_prompt = (
            "\nBelow are some functions from the repo you can also refer to:\n"
        )
        for idx, fn in enumerate(repo_coder_function_list):
            cur_str = f"""Function [{idx}]:
```Python
{fn['content']}
```
"""
            repo_coder_prompt += cur_str
        return repo_coder_prompt
