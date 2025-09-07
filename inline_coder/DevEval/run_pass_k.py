import os
import textwrap
from pathlib import Path
import psutil
from typing import Any, Union, Optional, Tuple, List
from rich import print
import func_timeout
import subprocess
from func_timeout import func_set_timeout
import numpy as np
from subprocess import run

from inline_coder.Preprocess.dev_eval_preprocess import DevEvalUtils
from inline_coder.utils.data_utils import (
    get_jsonl_data,
    save_json_data,
)
from inline_coder.Models.name_statics import ModelName
from inline_coder.utils.progress_manager import ProgressManager
from inline_coder.utils.logger_config import LoggerUtils
from inline_coder.DevEval.dev_eval_path_config import PathConfig

logger = LoggerUtils.get_main_logger(
    name=__name__, log_file="../LOGS/benchmark/DevEval/Pipeline.log"
)


class Execute:

    @classmethod
    def adjust_indent(cls, code, new_indent):
        dedented_code = textwrap.dedent(code)
        indented_code = textwrap.indent(dedented_code, " " * new_indent)
        return indented_code

    @classmethod
    def SetUp_evaluation(cls, data, completion):
        completion_path = Path(data["completion_path"])
        completion_path = os.path.join(DevEvalUtils.SourceCodePath, completion_path)
        head_tail = os.path.split(completion_path)
        completion_tmp_path = os.path.join(head_tail[0], "tmp_" + head_tail[1])
        run(["cp", completion_path, completion_tmp_path])
        sos, eos = data["body_position"][0] - 1, data["body_position"][1]
        with open(completion_path, "r") as f:
            file_lines = f.readlines()
        new_file_lines = file_lines[:sos] + ["\n", completion, "\n"] + file_lines[eos:]
        with open(completion_path, "w") as f:
            f.write("".join(new_file_lines))

    @classmethod
    @func_set_timeout(300)
    def execution_tests(cls, data):
        project_path = Path(
            os.path.join(DevEvalUtils.SourceCodePath, data["project_path"])
        ).resolve()
        project_path = str(project_path)
        command = ["python", "setup.py", "pytest", "--addopts"]

        for test in data["tests"]:
            process = subprocess.Popen(
                command + [test],
                cwd=project_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # break
            try:
                while True:
                    process_id = process.pid
                    process_memory = psutil.Process(process_id).memory_info().rss
                    if (
                        process_memory > 5 * 1024 * 1024 * 1024
                    ):  # 5GB memory usage per test
                        process.terminate()
                        process.wait()
                        return "OOM"  # Out of Memory
                    return_code = process.poll()
                    if return_code is not None:
                        if return_code != 0:
                            process.terminate()
                            process.wait()
                            print("Test Failed")
                            return "Error"  # Execution Error
                        else:
                            break
            except Exception as e:
                process.terminate()
                process.wait()
                print("Test ERROR")
                return "Error"  # Other Error
            finally:
                process.terminate()
                process.wait()
        return "Pass"  # Pass

    @classmethod
    def TearDown_evaluation(cls, data):
        completion_path = Path(data["completion_path"])
        completion_path = os.path.join(DevEvalUtils.SourceCodePath, completion_path)
        head_tail = os.path.split(completion_path)
        completion_tmp_path = os.path.join(head_tail[0], "tmp_" + head_tail[1])
        run(["mv", completion_tmp_path, completion_path])

    @classmethod
    def check_correctness(cls, origin_data, body_prediction):
        """
        Check the correctness of the execution result
        """
        flag_list = []
        if body_prediction.strip() == "pass":
            flag_list.append("Error")
        completion = cls.adjust_indent(body_prediction, origin_data["indent"])

        cls.SetUp_evaluation(origin_data, completion)
        try:
            flag = cls.execution_tests(origin_data)
        except func_timeout.exceptions.FunctionTimedOut:
            flag = "TimeOut"
        finally:

            cls.TearDown_evaluation(origin_data)
            # pass
        return flag

    @classmethod
    def Main(cls, args) -> Tuple[Path, Optional[dict]]:
        origin_data_path: Path = args.get("origin_data_path")
        assert origin_data_path is not None, "origin_data_path is required"
        generations_data_dir: Path = args.get("generations_dir")
        assert generations_data_dir is not None, "generations_data_dir is required"
        similarity_dir: Path = args.get("similarity_dir")
        assert similarity_dir is not None, "similarity_dir is required"
        save_dir: Path = args.get("save_dir")
        assert save_dir is not None, "save_dir is required"
        sample_num: int = args.get("sample_num", 1)
        limit: int = args.get("limit", -1)
        restart: bool = args.get("restart", False)

        original_data = get_jsonl_data(origin_data_path)
        generations_data = get_jsonl_data(generations_data_dir / "data.jsonl")
        similarity_data = get_jsonl_data(similarity_dir / "BLEU" / "data.jsonl")

        if limit > 0:
            original_data = original_data[:limit]
        new_data = []
        for idx in range(len(original_data)):
            new_sample = {
                "task_id": generations_data[idx]["task_id"],
                "origin": original_data[idx],
                "generations": generations_data[idx],
                "parsed_solution": similarity_data[idx]["parsed_solution"],
            }

            new_data.append(new_sample)
        save_dir.mkdir(parents=True, exist_ok=True)

        with ProgressManager(
            task_name="Execution",
            data_num=len(new_data),
            save_dir=save_dir,
            on_error=None,
            restart=restart,
            remove_existing=restart,
        ) as progress_manager:
            for i, new_sample in enumerate(new_data):
                if i < len(progress_manager.processed_data):
                    continue
                save_new_data = {
                    "task_id": i,
                    "parsed_solution": new_sample["parsed_solution"],
                    "execute_results": [],
                }
                for j in range(sample_num):
                    cur_sample_save = {}
                    for task_name in ["task_1_predictions", "task_2_predictions"]:
                        try:
                            cur_body = "\n".join(
                                new_sample["parsed_solution"][task_name][j][
                                    "response"
                                ].split("\n")[1:]
                            )
                            flag = cls.check_correctness(
                                new_sample["origin"],
                                cur_body,
                            )
                        except func_timeout.exceptions.FunctionTimedOut:
                            flag = "TimeOut"
                        except Exception as e:
                            logger.error(f"Error in execution: {e}")
                            flag = "Other Run Error"
                        cur_sample_save[task_name] = flag
                    save_new_data["execute_results"].append(cur_sample_save)
                progress_manager.update(advance=1, new_data=save_new_data)
        return save_dir, None


class PassATK:
    @classmethod
    def estimate_pass_at_k(
        cls,
        num_samples: Union[int, List[int], np.ndarray],
        num_correct: Union[List[int], np.ndarray],
        k: int,
    ) -> np.ndarray:
        """
        Estimates pass@k of each problem and returns them in an array.
        """

        def estimator(n: int, c: int, k: int) -> float:
            """
            Calculates 1 - comb(n - c, k) / comb(n, k).
            """
            if n - c < k:
                return 1.0
            return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))  # type: ignore

        if isinstance(num_samples, int):
            import itertools

            num_samples_it = itertools.repeat(num_samples, len(num_correct))
        else:
            assert len(num_samples) == len(num_correct)
            num_samples_it = iter(num_samples)

        return np.array(
            [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
        )

    @classmethod
    def Main(cls, args):
        origin_data_path: Path = args.get("origin_data_path")
        assert origin_data_path is not None, "origin_data_path is required"
        generations_data_dir: Path = args.get("generations_dir")
        assert generations_data_dir is not None, "generations_data_dir is required"
        similarity_dir: Path = args.get("similarity_dir")
        assert similarity_dir is not None, "similarity_dir is required"
        execute_dir: Path = args.get("save_dir")
        assert execute_dir is not None, "save_dir is required"
        sample_num: int = args.get("sample_num", 1)
        save_dir = execute_dir.parent / "pass_k"

        execute_result_path = execute_dir / "data.jsonl"
        execute_data = get_jsonl_data(execute_result_path)

        pass_at_k_compare = {}
        total, correct = [], []
        for task_name in ["task_1_predictions", "task_2_predictions"]:
            pass_at_k_compare[task_name] = {}
            cur_total, cur_correct = 0, 0
            for idx, sample in enumerate(execute_data):
                for j, execute_res in enumerate(sample["execute_results"]):
                    if execute_res[task_name] == "Pass":
                        cur_correct += 1
                    cur_total += 1
            total.append(cur_total)
            correct.append(cur_correct)

            for cur_k in [1, 2]:
                if cur_k > sample_num:
                    continue
                pass_at_k_list = cls.estimate_pass_at_k(
                    num_samples=total,
                    num_correct=correct,
                    k=cur_k,
                )

                pass_at_k_compare[task_name][f"Pass@{cur_k}"] = pass_at_k_list.mean()

        print(pass_at_k_compare)
        save_json_data(pass_at_k_compare, save_dir / "pass_at_k.json")


class Pipeline:
    FilteredDatasetPath = Path("../data/Benchmark/dev_eval") / "test" / "data.jsonl"
    BenchMarkName = "dev_eval"
    BenchMarkDir = Path("../data/Benchmark") / BenchMarkName

    @classmethod
    def PassKPipeline(cls):
        """Use a little data to the pipelien workflow"""
        model = ModelName.GPT5
        model_name = model.value
        generation_dir = (
            PathConfig.BenchmarkDir
            / "main_table_v1"
            / "total_gen"
            / model_name
            / "generations"
        )
        similarity_dir = generation_dir.parent / "similarity"
        limit = -1
        sample_num = 1
        save_dir = generation_dir.parent / "execute_results"

        args = {
            "origin_data_path": PathConfig.OriginDataPath,
            "generations_dir": generation_dir,
            "similarity_dir": similarity_dir,
            "save_dir": save_dir,
            "sub_dir_name": save_dir,
            "sample_num": sample_num,
            "limit": limit,
            "restart": True,
        }
        Execute.Main(args)
        PassATK.Main(args)


if __name__ == "__main__":
    Pipeline.PassKPipeline()
