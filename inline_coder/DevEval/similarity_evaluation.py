from pathlib import Path
from typing import Any, DefaultDict, Optional
import numpy as np

from inline_coder.utils.logger_config import LoggerUtils

logger = LoggerUtils.get_main_logger(
    name=__name__, log_file="LOGS/benchmark/DevEval/similarity_evaluation.log"
)
from inline_coder.Evaluation.EM_evaluator import EMEvaluator
from inline_coder.Evaluation.ES_evaluator import ESEvaluator
from inline_coder.Evaluation.BLEU_evaluator import BLEUEvaluator
from inline_coder.Preprocess.dev_eval_preprocess import DevEvalUtils
from inline_coder.utils.data_utils import (
    get_jsonl_data,
    save_json_data,
    save_text_data,
)
from inline_coder.utils.progress_manager import ProgressManager
from inline_coder.ResultProcess.result_process import ResultProcess
from inline_coder.utils.Display import TableMaker

from inline_coder.Evaluation.evaluate_utils import (
    BLEUProcess,
    EMProcess,
    ESProcess,
    MetricProcess,
)


class ListResEvaluation:
    BenchMarkName = "repo_exec"
    BenchMarkDir = Path("data/Benchmark") / BenchMarkName
    DevEvalBenckMark = Path("data/Benchmark") / "dev_eval"
    metric_process_list: list[type[MetricProcess]] = [
        BLEUProcess,
        EMProcess,
        ESProcess,
    ]

    @classmethod
    def parse_generations(cls, generation_data, original_data):
        prediction_num = 0
        error_count_503 = 0
        new_data = []
        for idx, sample in enumerate(generation_data):
            origin_sample = original_data[idx]
            DevEvalUtils.attach_details_from_file_for_sample(origin_sample)
            new_sample = {
                "task_id": sample["task_id"],
                "generations": sample,
                "origin": origin_sample,
            }
            parsed_lists = []
            for prompts, predicts in zip(
                sample["prompt_lists"], sample["prediction_lists"]
            ):
                task_parsed = []
                for prompt, pred in zip(prompts, predicts):
                    prediction_num += 1
                    if "Error code: 503" in pred["response"]:
                        error_count_503 += 1
                        continue
                    parsed_response = ResultProcess.common_parse(
                        target_signature=new_sample["origin"]["target_signature"],
                        gen_prediction=pred["response"],
                    )
                    task_parsed.append(
                        {
                            "prompt": prompt,
                            "response": parsed_response,
                        }
                    )
                parsed_lists.append(task_parsed)
            new_sample["parsed_solution"] = {
                "reference": new_sample["origin"]["solution"],
                "generations": parsed_lists,
            }
            new_data.append(new_sample)
        result = {
            "prediction_num": prediction_num,
            "error_count_503": error_count_503,
            "new_data": new_data,
        }
        return result

    @classmethod
    def CalculateSimilarity(cls, args: dict):
        origin_data_path: Optional[Path] = args.get("origin_data_path")
        assert origin_data_path is not None
        generations_dir: Optional[Path] = args.get("generations_dir")
        assert generations_dir is not None
        save_dir: Optional[Path] = args.get("save_dir")
        assert save_dir is not None
        restart: bool = args.get("restart", False)

        original_data = get_jsonl_data(origin_data_path)
        generation_data = get_jsonl_data(generations_dir / "data.jsonl")

        # attach original data to processed data
        rich_data = []
        total_predictions = 0
        error_count_503 = 0

        parsed_result = cls.parse_generations(generation_data, original_data)
        total_predictions = parsed_result["prediction_num"]
        error_count_503 = parsed_result["error_count_503"]
        rich_data = parsed_result["new_data"]
        print(
            f"[bold red]total generations:{len(generation_data)}, total predictions:{total_predictions}, 503 error count:{error_count_503}[/]"
        )

        metric_args = {}
        for metric_process in cls.metric_process_list:
            with ProgressManager(
                task_name=f"Similarity Calculate for {metric_process.NAME}",
                data_num=len(rich_data),
                save_dir=save_dir / metric_process.NAME,
                on_error=None,
                restart=restart,
                remove_existing=restart,
            ) as progress_manager:
                for idx, sample in enumerate(rich_data):
                    if idx < len(progress_manager.processed_data):
                        continue
                    score_result = metric_process.process(
                        sample["parsed_solution"], metric_args
                    )
                    new_data = {
                        "task_id": sample["task_id"],
                        "project": sample["origin"]["project_path"],
                        "completion_path": sample["origin"]["completion_path"],
                        "namespace": sample["origin"]["namespace"],
                        "generations": sample["generations"],
                        "parsed_solution": sample["parsed_solution"],
                        "solution": sample["origin"]["solution"],
                        "similarity_scores": score_result,
                    }
                    progress_manager.update(
                        advance=1,
                        new_data=new_data,
                    )

                progress_manager.info["error_info"] = {}
                progress_manager.info["error_info"]["503_error_count"] = error_count_503
                progress_manager.info["error_info"][
                    "total_predictions_num"
                ] = total_predictions

    @classmethod
    def GetAverageScore(cls, args: dict):
        similarity_dir: Path = args.get("save_dir")  # type: ignore
        score_map = {}

        for metric_process in cls.metric_process_list:
            metric_name = metric_process.NAME
            score_map[metric_name] = []
            scores_data = get_jsonl_data(
                Path(similarity_dir) / metric_name / "data.jsonl"
            )
            for sample in scores_data:
                for sc_list in sample["similarity_scores"]:
                    score_map[metric_name].append(sc_list)

        average_scores = {}
        for metric_process in cls.metric_process_list:
            metric_name = metric_process.NAME
            average_scores[metric_name] = []
            score_array = np.array(score_map[metric_name])
            for i in range(score_array.shape[1]):
                average_scores[metric_name].append(float(np.mean(score_array[:, i])))

        save_json_data(average_scores, Path(similarity_dir) / "average.json")
        head_line = ["GroupName"] + [mc_pr.NAME for mc_pr in cls.metric_process_list]
        data_lines = []
        for i in range(len(list(average_scores.values())[0])):
            if i == 0:
                line = ["Base Gen"]
            else:
                line = [f"Iteration_{i}"]
            for metric_process in cls.metric_process_list:
                metric_name = metric_process.NAME
                line.append(f"{average_scores[metric_name][i]:.4f}")
            data_lines.append(line)
        table_txt = TableMaker.make_table(head_line=head_line, data_lines=data_lines)
        save_text_data(table_txt, Path(similarity_dir) / "average_table.md")
        return score_map


class Similarity:

    @staticmethod
    def bleu_process(
        sample: dict[str, Any], args: dict[str, Any]
    ) -> dict[str, list[float]]:
        result = {}
        sample_num = args["sample_num"]
        reference = sample["parsed_solution"]["reference"]
        if args["task_1"]:
            task_1_scores = []
            for predict in sample["parsed_solution"]["task_1_predictions"][:sample_num]:
                task_1_scores.append(
                    BLEUEvaluator.evaluate_one_pair(
                        predict=predict["response"],
                        reference=reference,
                    )
                )
            result["task_1_scores"] = task_1_scores

        if args["task_2"]:

            task_2_scores = []
            for predict in sample["parsed_solution"]["task_2_predictions"][:sample_num]:

                current_predict = predict["response"]
                # try to remove """""" in python to get better bleu score
                clean_predict = ResultProcess.clean_multi_line_comment(current_predict)
                score_v_0 = BLEUEvaluator.evaluate_one_pair(
                    predict=current_predict,
                    reference=reference,
                )
                score_v_clean = BLEUEvaluator.evaluate_one_pair(
                    predict=clean_predict,
                    reference=reference,
                )
                if score_v_clean > score_v_0:
                    predict["response"] = clean_predict
                task_2_scores.append(max(score_v_0, score_v_clean))
            result["task_2_scores"] = task_2_scores
        return result

    @staticmethod
    def em_process(
        sample: dict[str, Any], args: dict[str, Any]
    ) -> dict[str, list[float]]:
        result = {}
        sample_num = args["sample_num"]
        reference = sample["parsed_solution"]["reference"]
        if args["task_1"]:
            task_1_scores = []
            for predict in sample["parsed_solution"]["task_1_predictions"][:sample_num]:
                task_1_scores.append(
                    EMEvaluator.evaluate_one_pair(
                        predict=predict["response"],
                        reference=reference,
                    )
                )
            result["task_1_scores"] = task_1_scores
        if args["task_2"]:
            task_2_scores = []
            for predict in sample["parsed_solution"]["task_2_predictions"][:sample_num]:
                task_2_scores.append(
                    EMEvaluator.evaluate_one_pair(
                        predict=predict["response"],
                        reference=reference,
                    )
                )
            result["task_2_scores"] = task_2_scores
        return result

    @staticmethod
    def es_process(
        sample: dict[str, Any], args: dict[str, Any]
    ) -> dict[str, list[float]]:
        result = {}
        sample_num = args["sample_num"]
        reference = sample["parsed_solution"]["reference"]
        if args["task_1"]:
            task_1_scores = []
            for predict in sample["parsed_solution"]["task_1_predictions"][:sample_num]:
                task_1_scores.append(
                    ESEvaluator.evaluate_one_pair(
                        predict=predict["response"],
                        reference=reference,
                    )
                )
            result["task_1_scores"] = task_1_scores
        if args["task_2"]:
            task_2_scores = []
            for predict in sample["parsed_solution"]["task_2_predictions"][:sample_num]:
                task_2_scores.append(
                    ESEvaluator.evaluate_one_pair(
                        predict=predict["response"],
                        reference=reference,
                    )
                )
            result["task_2_scores"] = task_2_scores
        return result

    metric_list = [
        (BLEUEvaluator.NAME, bleu_process),
        (EMEvaluator.NAME, em_process),
        (ESEvaluator.NAME, es_process),
    ]

    @classmethod
    def parse_generations(cls, generation_data, original_data):
        new_data = []
        total_predictions = 0
        error_count_503 = 0
        for idx, sample in enumerate(generation_data):
            new_sample = {
                "task_id": sample["task_id"],
                "generations": sample,
                "origin": original_data[idx],
            }
            sample["origin"] = original_data[idx]
            body_range = sample["origin"]["body_position"]

            # 补全可能缺失的提取信息
            if "target_file" not in sample["origin"]:
                target_file = (
                    Path(DevEvalUtils.SourceCodePath)
                    / sample["origin"]["completion_path"]
                )
                with open(target_file, "r") as f:
                    sample["origin"]["target_file"] = f.read()
                # get target_signature
                file_lines = sample["origin"]["target_file"].split("\n")
                signature_lines = file_lines[
                    sample["origin"]["signature_position"][0]
                    - 1 : sample["origin"]["signature_position"][1]
                ]
                sample["origin"]["target_signature"] = "\n".join(signature_lines)

            # get reference body
            target_file_lines = sample["origin"]["target_file"].split("\n")
            target_body = target_file_lines[body_range[0] - 1 : body_range[1]]
            sample["reference_body"] = "\n".join(target_body)

            task_1_parsed, task_2_parsed = [], []
            for prompt, pred in zip(
                sample["task_1_prompts"], sample["task_1_predictions"]
            ):
                total_predictions += 1
                if "Error code: 503" in pred["response"]:
                    error_count_503 += 1
                    continue
                task_1_parsed.append(
                    {
                        "prompt": prompt,
                        "response": ResultProcess.clean_and_get_function(
                            sample["origin"]["target_signature"], pred["response"]
                        ),
                    }
                )
            for prompt, pred in zip(sample["prompts"], sample["predictions"]):
                total_predictions += 1
                if "Error code: 503" in pred["response"]:
                    error_count_503 += 1
                    continue
                task_2_parsed.append(
                    {
                        "prompt": prompt,
                        "response": ResultProcess.clean_and_get_function(
                            sample["origin"]["target_signature"], pred["response"]
                        ),
                    }
                )
            for task_1_parse, task_2_parse in zip(task_1_parsed, task_2_parsed):
                task_2_response_lines = task_2_parse["response"].split("\n")
                if len(task_2_response_lines) == 2:
                    if task_2_response_lines[1].strip() == "pass":
                        task_2_parse["response"] = task_1_parse["response"]
            new_sample["parsed_solution"] = {
                "reference": sample["origin"]["target_signature"]
                + "\n"
                + sample["reference_body"],
                "reference_body": sample["reference_body"],
                "task_1_predictions": task_1_parsed,
                "task_2_predictions": task_2_parsed,
            }
            new_data.append(new_sample)
        return new_data

    @classmethod
    def CalculateSimilarity(
        cls,
        origin_data_path: Path,
        generations_dir: Path,
        save_dir: Path,
        restart: bool,
    ):
        original_data = get_jsonl_data(origin_data_path)
        generation_data = get_jsonl_data(generations_dir / "data.jsonl")
        rich_data = cls.parse_generations(generation_data, original_data)

        metric_args = {
            "task_1": True,
            "task_2": True,
            "sample_num": 2,
        }
        for metric_name, process_func in cls.metric_list:
            with ProgressManager(
                task_name=f"Similarity Calculate for {metric_name}",
                data_num=len(rich_data),
                save_dir=save_dir / metric_name,
                on_error=None,
                restart=restart,
                remove_existing=restart,
            ) as progress_manager:
                for idx, sample in enumerate(rich_data):
                    if idx < len(progress_manager.processed_data):
                        continue
                    score_result = process_func(sample, metric_args)
                    new_data = {
                        "task_id": sample["task_id"],
                        "namespace": sample["origin"]["namespace"],
                        "predictions": sample["generations"],
                        "parsed_solution": sample["parsed_solution"],
                        "solution": sample["parsed_solution"]["reference"],
                        "similarity_scores": score_result,
                    }
                    progress_manager.update(
                        advance=1,
                        new_data=new_data,
                    )

    @classmethod
    def GetAverageScore(cls, save_dir: Path):
        score_map = {
            "task_1": DefaultDict(float),
            "task_2": DefaultDict(float),
        }

        for metric_name, _ in cls.metric_list:
            scores_data = get_jsonl_data(Path(save_dir) / metric_name / "data.jsonl")

            for task_num in [1, 2]:
                sample_average_list = []
                for sample in scores_data:
                    sample_average_list.append(
                        np.mean(
                            np.array(
                                sample["similarity_scores"][f"task_{task_num}_scores"]
                            )
                        )
                    )
                score_map[f"task_{task_num}"][metric_name] = float(
                    np.mean(np.array(sample_average_list))
                )
        save_json_data(score_map, Path(save_dir) / "average_score.json")
        return (Path(save_dir) / "data.jsonl", None)
