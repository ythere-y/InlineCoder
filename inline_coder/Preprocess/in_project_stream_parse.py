import signal
from collections import defaultdict
from itertools import groupby
from pydepcall.extractor import Extractor
from pydepcall.Node import FunctionNode, ModuleNode
from datasets import Dataset
from pathlib import Path
from typing import List, Dict, Any
from rich import print
from tqdm import tqdm

from inline_coder.utils.progress_manager import ProgressManager


class TimeoutException(Exception):
    pass


def handler(signum, frame):
    raise TimeoutException("Function execution timed out")


def find_updown_stream(
    reposrc: Path, function_name_map: Dict[str, List[Dict[str, Any]]]
):
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(60)  # 设置超时时间为60秒
    try:
        extractor = Extractor(str(reposrc))
        print(f"[bold red]Ready to extract from {reposrc}[/]")
        output = extractor.extract()
        assert isinstance(output, dict)
    except TimeoutException as te:
        print(f"Timeout while processing {reposrc}: {te}")
        return
    except Exception as e:
        print(f"Error processing {reposrc}: {e}")
        return
    finally:
        signal.alarm(0)
    # 遍历每个函数，找到调用边
    for file_name, file_data in output.items():
        file_data: ModuleNode
        for function_info in file_data.function_list:
            function_info: FunctionNode
            if function_info.name in function_name_map:
                # 加入calling相关下游信息
                for child in function_info.children:
                    if not type(child) == FunctionNode:
                        continue
                    for sample in function_name_map[function_info.name]:
                        current_calling_edge = {
                            "path": child.path,
                            "name": child.name,
                            "content": child.content,
                            "docstring": child.docstring,
                        }
                        sample["downstream"].append(current_calling_edge)
            for child in function_info.children:
                if not type(child) == FunctionNode:
                    continue
                # 如果这个函数是我们需要的entry_point
                if child.name in function_name_map:
                    # 遍历entry_point_map，找到对应的样本
                    for sample in function_name_map[child.name]:
                        # 添加调用边信息
                        current_calling_edge = {
                            # "caller_info": function_info,
                            "path": function_info.path,
                            "name": function_info.name,
                            "content": function_info.content,
                            "docstring": function_info.docstring,
                        }
                        sample["upstream"].append(current_calling_edge)


def extract_find_functions(reposrc: Path):
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(60)  # 设置超时时间为60秒
    try:
        extractor = Extractor(str(reposrc))
        print(f"[bold red]Ready to extract from {reposrc}[/]")
        output = extractor.extract()
        assert isinstance(output, dict)
    except TimeoutException as te:
        print(f"Timeout while processing {reposrc}: {te}")
        return
    except Exception as e:
        print(f"Error processing {reposrc}: {e}")
        return
    finally:
        signal.alarm(0)
    functions = []
    for file_name, file_data in output.items():
        file_data: ModuleNode
        for function_info in file_data.function_list:
            function_info: FunctionNode
            functions.append(
                {
                    "path": file_name,
                    "name": function_info.name,
                    "content": function_info.content,
                    "docstring": function_info.docstring,
                }
            )
    return functions


def extract_and_save(dataset, project_path_key: str, save_dir: Path):
    reposrc_list = list(set([sample[project_path_key] for sample in dataset]))
    with ProgressManager(
        task_name="extract_and_save",
        data_num=len(reposrc_list),
        save_dir=save_dir,
        restart=False,
    ) as progress:
        idx = 0
        for reposrc in tqdm(reposrc_list, desc="Processing repositories"):
            if idx < len(progress.processed_data):
                idx += 1
                continue
            functions = extract_find_functions(reposrc)
            cur_data = {
                "project_path": reposrc,
                "functions": functions,
            }
            progress.update(advance=1, new_data=cur_data)

            idx += 1

    return save_dir / "data.jsonl"


def attach_streaming_info(
    dataset,
    project_path_key: str,
    function_name_key: str,
    save_dir: Path,
) -> Dataset:
    total_data_len = len(dataset)
    with ProgressManager(
        task_name="group_attach_streaming_info",
        data_num=total_data_len,
        save_dir=save_dir,
        restart=False,
        # save_as_pkl=True,
    ) as progress:
        group_result = groupby(
            dataset,
            key=lambda x: x[project_path_key],
        )
        idx = 0
        for key, group in group_result:
            group = list(group)
            if idx < len(progress.processed_data):
                idx += len(group)
                continue
            for sample in group:
                assert isinstance(sample, dict)
                sample["upstream"] = []
                sample["downstream"] = []

            grouped_data = []
            function_name_map = defaultdict(list)
            for sample in group:
                assert isinstance(sample, dict)
                entry_point = sample[function_name_key]
                function_name_map[entry_point].append(sample)

            # 解析项目,并且检索上下游信息
            reposrc = key
            find_updown_stream(reposrc, function_name_map)

            for sample in group:
                grouped_data.append(sample)
            for cur_data in grouped_data:
                progress.update(advance=1, new_data=cur_data)
                idx += 1
        result_data: List[Dict[str, Any]] = progress.processed_data
    result_dataset = Dataset.from_list(result_data)
    return result_dataset
