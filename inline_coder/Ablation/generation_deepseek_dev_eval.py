import os
import sys
import fire
from tqdm import tqdm
from enum import Enum
from rich import print
from pathlib import Path
from typing import Union, Optional
from dataclasses import dataclass

sys.path.append(os.getcwd())
from utils.logger_config import LoggerUtils

logger = LoggerUtils.get_main_logger(
    name=__name__,
    log_file="LOGS/Ablation/background_generation.log",
)
from inline_coder.Preprocess.dev_eval_preprocess import DevEvalUtils
from inline_coder.Models.name_statics import ModelName
from inline_coder.Evaluation.PPL_evaluator import PPLEvaluator
from inline_coder.Prompt.dev_eval_prompt_builder import DevEvalPromptBuilder
from inline_coder.utils.data_utils import get_jsonl_data
from inline_coder.utils.decorators import Decorator
from inline_coder.ResultProcess.result_process import DownStreamAttach, ResultProcess
from inline_coder.Models.services import deepseek_service
from inline_coder.utils.progress_manager import ProgressManager
from inline_coder.Parser.python_inline import PythonParser as PythonInlineParser
from inline_coder.DevEval.similarity_evaluation import ListResEvaluation


@dataclass
class AblationArgs:
    no_upstream: bool = False
    no_inline: bool = False
    no_downstream: bool = False
    no_ppl: bool = False
    no_solution: bool = False

    def get_ablation_str(self) -> str:
        ablation_parts = []
        if self.no_upstream:
            ablation_parts.append("no_upstream")
        if self.no_inline:
            ablation_parts.append("no_inline")
        if self.no_downstream:
            ablation_parts.append("no_downstream")
        if self.no_ppl:
            ablation_parts.append("no_ppl")
        if self.no_solution:
            ablation_parts.append("no_solution")
        if not ablation_parts:
            return "full_model"
        return "_".join(ablation_parts)


class DataName(Enum):
    DevEval = "DevEval"


class DataUtils:
    BenchmarkDir = None

    @classmethod
    def check_sample(cls, sample: dict):
        raise NotImplementedError

    @classmethod
    def BasePrompt(cls, sample: dict) -> str:
        raise NotImplementedError

    @classmethod
    def WithDownStreamPrompt(cls, sample: dict) -> str:
        raise NotImplementedError

    @classmethod
    def UpDownInjectPrompt(
        cls,
        sample: dict,
        inlined_result_list: list,
        useful_downstream: list,
        solution: str,
        ppl_result: Optional[dict] = None,
    ) -> str:
        raise NotImplementedError

    @classmethod
    def get_detail_info(cls, sample: dict) -> dict:
        raise NotImplementedError


class DevEvalDataUtils(DataUtils):
    BenchmarkDir = Path("data/Benchmark/dev_eval")

    @classmethod
    def check_sample(cls, sample: dict):
        sample = DevEvalUtils.attach_details_from_file_for_sample(sample)
        return

    @classmethod
    def BasePrompt(cls, sample: dict) -> str:
        base_prompt = DevEvalPromptBuilder.BasePrompt(sample)
        return base_prompt

    @classmethod
    def WithDownStreamPrompt(cls, sample: dict) -> str:
        base_prompt = DevEvalPromptBuilder.DevEvalBasePromptWithDownStream(sample)
        return base_prompt

    @classmethod
    def UpDownInjectPrompt(
        cls,
        sample: dict,
        inlined_result_list: list,
        useful_downstream: list,
        solution: str,
        ppl_result: Optional[dict] = None,
    ) -> str:
        prompt = DevEvalPromptBuilder.UpDownInjectPrompt(
            sample, inlined_result_list, useful_downstream, solution, ppl_result
        )
        return prompt

    @classmethod
    def get_detail_info(cls, sample: dict) -> dict:
        return {
            "target_signature": sample["target_signature"],
            "target_project_path": sample["target_project_path"],
            "target_function_name": sample["target_function_name"],
            "upstream": sample["upstream"],
        }


class Generation:
    service_func = deepseek_service
    data_utils = DevEvalDataUtils
    BenchmarkDir = data_utils.BenchmarkDir

    @classmethod
    def init(cls, data: DataName, model: ModelName):
        if data == DataName.DevEval:
            cls.data_utils = DevEvalDataUtils
            cls.BenchmarkDir = cls.data_utils.BenchmarkDir
        else:
            raise ValueError(f"Unsupported data name: {data}")

        if model == ModelName.DeepSeekV3:
            cls.service_func = deepseek_service
        else:
            raise ValueError(f"Unsupported model name: {model}")

    @classmethod
    @Decorator.report
    @Decorator.with_naming
    def AblationGeneration(cls, args: dict) -> tuple[Union[str, Path], Optional[dict]]:
        """
        Voke generation using OpenAI API
        """
        data_path = args.get("data_path")
        assert data_path is not None
        limit = args.get("limit", -1)
        save_dir: Optional[Path] = args.get("save_dir")
        assert save_dir is not None, "save_dir is None"
        online = args.get("online", False)
        sample_num = args.get("sample_num", 1)
        restart = args.get("restart", False)
        albation_args: AblationArgs = args.get("ablation_args", AblationArgs())
        based_on_existing = args.get("based_on_existing", None)

        dataset = get_jsonl_data(data_path)
        if limit > 0:
            dataset = dataset[:limit]

        exist_generation_data = []
        if based_on_existing is not None and based_on_existing.get("switch", True):
            logger.info(
                f"Based on existing generations from {based_on_existing.get('exist_generation_dir', None)}"
            )
            exist_generation_dir: Optional[Path] = based_on_existing.get(
                "exist_generation_dir", None
            )
            assert exist_generation_dir is not None
            exist_generation_data = get_jsonl_data(exist_generation_dir / "data.jsonl")
            exist_generation_data = exist_generation_data[: len(dataset)]

        extracted_data = []
        extracted_data = get_jsonl_data(
            cls.BenchmarkDir / "extract_save" / "data.jsonl"
        )
        with ProgressManager(
            task_name="API LLM Generation",
            data_num=len(dataset),
            save_dir=save_dir,
            on_error=None,
            restart=restart,
            remove_existing=restart,
        ) as progress_manager:
            for i, sample in enumerate(dataset):

                if i < len(progress_manager.processed_data):
                    continue
                cls.data_utils.check_sample(sample)
                new_data = {
                    "task_id": i,
                    "prompt_lists": list(),
                    "prediction_lists": list(),
                }
                matched_functions = []

                detail_info = cls.data_utils.get_detail_info(sample)
                target_signature = detail_info["target_signature"]
                project_path = detail_info["target_project_path"]
                target_function_name = detail_info["target_function_name"]
                upstream = detail_info["upstream"]
                if albation_args.no_downstream:
                    base_prompt = cls.data_utils.BasePrompt(sample)
                else:
                    base_prompt = cls.data_utils.WithDownStreamPrompt(sample)
                    for extracted_sample in extracted_data:
                        if extracted_sample["project_path"] == project_path:
                            matched_functions = extracted_sample["functions"]
                            if matched_functions is None:
                                matched_functions = []
                            break

                for j in tqdm(range(sample_num), desc="Batch Generate for one sample"):
                    if based_on_existing is not None and based_on_existing.get(
                        "switch", True
                    ):
                        if "task_1_predictions" in exist_generation_data[i]:
                            base_response = exist_generation_data[i][
                                "task_1_predictions"
                            ][j]

                        else:
                            base_response = exist_generation_data[i][
                                "prediction_lists"
                            ][j]
                    else:
                        base_response = cls.service_func(base_prompt, online)

                    current_prompt_list = [base_prompt]
                    current_prediction_list = [base_response]

                    result_info = ResultProcess.with_downstream_parse(
                        target_signature=target_signature,
                        gen_prediction=base_response["response"],
                    )
                    solution_fn = result_info["target_function"]

                    if not albation_args.no_downstream:
                        matched_functions = []
                        for extracted_sample in extracted_data:
                            if extracted_sample["project_path"] == project_path:
                                matched_functions = extracted_sample["functions"]
                                if matched_functions is None:
                                    matched_functions = []
                                break
                        useful_downstream = DownStreamAttach.GetUsefuleDownStreams(
                            target_function_name=target_function_name,
                            function=solution_fn,
                            ai_pred_downstream=result_info["down_stream"],
                            candidate_functions=matched_functions,
                        )
                    else:
                        useful_downstream = []

                    if not albation_args.no_ppl:
                        full_text = base_prompt + base_response["response"]
                        ppl_result = PPLEvaluator.get_ppl(
                            text={
                                "full_text": full_text,
                                "prefix_text": base_prompt,
                            },
                            condition_mode="prefix",
                        )
                    else:
                        ppl_result = None

                    if not albation_args.no_upstream:
                        inlined_result_list = list()
                        caller_info_list = []
                        if not albation_args.no_inline:
                            for up_func in upstream:
                                call_edge_info = {
                                    "caller": up_func["content"],
                                    "callee": solution_fn,
                                }
                                caller_info_list.append(call_edge_info)
                                inlined_result = PythonInlineParser.InlineMethod(
                                    call_edge_info
                                )
                                inlined_result_list.append(inlined_result)
                        else:
                            for up_func in upstream:
                                no_inline_info = {
                                    "result": up_func["content"],
                                    "edit_lines": [],
                                    "edit_codes": [],
                                }
                                inlined_result_list.append(no_inline_info)
                    else:
                        inlined_result_list = []
                    if albation_args.no_solution:
                        solution_fn = ""
                    inject_prompt = cls.data_utils.UpDownInjectPrompt(
                        sample,
                        inlined_result_list,
                        useful_downstream,
                        solution_fn,
                        ppl_result,
                    )

                    inject_response = cls.service_func(inject_prompt, online)

                    current_prompt_list.append(inject_prompt)
                    current_prediction_list.append(inject_response)

                    new_data["prompt_lists"].append(current_prompt_list)
                    new_data["prediction_lists"].append(current_prediction_list)

                progress_manager.update(advance=1, new_data=new_data)
        return save_dir, None


def run_ablation(
    ablation_type="no_line",
    model="DeepSeekV3",
    dataset="DevEval",
    logger_log_path="LOGS/Ablation/background/deepseek_dev_eval.log",
):
    global logger
    logger = LoggerUtils.get_main_logger(
        name=__name__,
        log_file=logger_log_path,
    )
    ablation_map = {
        "no_upstream": AblationArgs(no_upstream=True),
        "no_inline": AblationArgs(no_inline=True),
        "no_downstream": AblationArgs(no_downstream=True),
        "no_ppl": AblationArgs(no_ppl=True),
        "no_solution": AblationArgs(no_solution=True),
        "full_model": AblationArgs(),
    }
    ablation_lists = [ablation_map[ablation_type]]
    model_map = {
        "DeepSeekV3": ModelName.DeepSeekV3,
    }
    model_list = [model_map[model]]

    data_map = {
        "DevEval": DataName.DevEval,
    }
    data_list = [data_map[dataset]]
    for ablation_args in ablation_lists:
        for model in model_list:
            for dataset in data_list:
                print(
                    f"Start ablation: {ablation_args}, model: {model}, dataset: {dataset}"
                )
                Generation.init(dataset, model)
                model_name = model.value
                origin_data_path = (
                    Generation.BenchmarkDir / "new_clean_data" / "data.jsonl"
                )
                generation_dis = (
                    Generation.BenchmarkDir
                    / "ablation"
                    / ablation_args.get_ablation_str()
                    / model_name
                    / "generations"
                )
                base_generation_dir = (
                    Generation.BenchmarkDir
                    / "main_table_v1"
                    / "total_gen"
                    / model_name
                    / "generations"
                )
                args = {
                    "data_path": origin_data_path,
                    "limit": -1,
                    "save_dir": generation_dis,
                    "online": True,
                    "sample_num": 1,
                    "restart": False,
                    "ablation_args": ablation_args,
                    "based_on_existing": {
                        "switch": True,
                        "exist_generation_dir": base_generation_dir,
                    },
                }
                Generation.AblationGeneration(args)
                similarity_dir = generation_dis.parent / "similarity"
                evaluate_args = {
                    "origin_data_path": origin_data_path,
                    "generations_dir": generation_dis,
                    "save_dir": similarity_dir,
                    "restart": True,
                }
                ListResEvaluation.CalculateSimilarity(evaluate_args)
                ListResEvaluation.GetAverageScore(evaluate_args)


if __name__ == "__main__":
    fire.Fire(run_ablation)
