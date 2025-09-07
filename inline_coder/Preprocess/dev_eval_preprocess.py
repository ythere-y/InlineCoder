import jsonlines
import os
from rich import print
import tree_sitter_python as tspy
from tree_sitter import Language, Parser
from pathlib import Path
from inline_coder.utils import logger_config

logger = logger_config.LoggerUtils.get_main_logger(
    name="load_dev_eval", log_file="LOGS/load_dev_eval.log"
)
from inline_coder.Preprocess.in_project_stream_parse import (
    extract_and_save,
    attach_streaming_info,
)


class PaserLoader:
    PYTHON_LANGUAGE = tspy.language()
    python_paser = Parser(Language(tspy.language()))


class DevEvalUtils:
    BenchMarkDir = Path("data/Benchmark/dev_eval")
    data_path = "references/DevEval/data.jsonl"
    SourceCodePath = "references/DevEval/Source_Code"
    DependencyPath = "references/DevEval/Dependency_Code"

    PromptDir = "references/DevEval/Experiments/prompt"

    @classmethod
    def __get_pure_data(cls):
        """get pure data from dev eval dataset"""
        data = []
        with jsonlines.open(cls.data_path) as reader:
            for obj in reader:
                data.append(obj)
        return data

    @classmethod
    def __attach_target_function_name(cls, dataset):
        """为每个样本添加目标函数的名字"""
        for sample in dataset:
            target_function_name = sample["namespace"].split(".")[-1]
            sample["target_function_name"] = target_function_name
        return dataset

    @classmethod
    def __attach_target_project_path(cls, dataset):
        """为每个样本添加目标项目的文件路径（绝对路径）"""
        for sample in dataset:
            project_path = sample["project_path"]
            target_project_path = os.path.join(cls.SourceCodePath, project_path)
            sample["target_project_path"] = target_project_path
        return dataset

    @classmethod
    def PreprocessData(cls):
        """
        Get processed data from original dev eval dataset.
        1.add target function name
        2.add target project absolute path
        3.add upstream and downstream calling info
        """
        data = cls.__get_pure_data()
        data = cls.__attach_target_function_name(data)
        data = cls.__attach_target_project_path(data)
        data = attach_streaming_info(
            data,
            project_path_key="target_project_path",
            function_name_key="target_function_name",
            save_dir=Path("data/Benchmark/dev_eval/processed_clean_data"),
        )
        return data

    @classmethod
    def ExtractSave(cls):
        """
        extract functions info from all projects and save them together
        """
        data = cls.__get_pure_data()
        data = cls.__attach_target_function_name(data)
        data = cls.__attach_target_project_path(data)

        save_path = extract_and_save(
            data,
            project_path_key="target_project_path",
            save_dir=Path("data/Benchmark/dev_eval/extract_save"),
        )
        print(f"saved into :{save_path}")

    @classmethod
    def attach_details_from_file_for_sample(cls, sample):
        if "target_file" not in sample:
            target_file_path = os.path.join(
                sample["target_project_path"],
                "/".join(sample["completion_path"].split("/")[2:]),
            )
            with open(target_file_path, "r", encoding="utf-8") as f:
                target_file_content = f.read()
            sample["target_file"] = target_file_content
            file_lines = target_file_content.split("\n")
            signature_position = sample["signature_position"]
            signature_str = "\n".join(
                file_lines[signature_position[0] - 1 : signature_position[1]]
            )
            sample["target_signature"] = signature_str
            body_position = sample["body_position"]
            body_str = "\n".join(file_lines[body_position[0] - 1 : body_position[1]])
            sample["solution"] = signature_str + "\n" + body_str
        return sample


if __name__ == "__main__":
    data = DevEvalUtils.PreprocessData()
    DevEvalUtils.ExtractSave()
