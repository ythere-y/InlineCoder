import os
import sys

sys.path.append(os.getcwd())
if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from pathlib import Path
from typing import Union, Optional
from tqdm import tqdm
from rich import print
from inline_coder.utils.logger_config import LoggerUtils
from inline_coder.DevEval.dev_eval_path_config import PathConfig
from inline_coder.DevEval.similarity_evaluation import Similarity
from inline_coder.Evaluation.PPL_evaluator import PPLEvaluator
from inline_coder.Models.name_statics import ModelName
from inline_coder.Prompt.dev_eval_prompt_builder import DevEvalPromptBuilder
from inline_coder.utils.data_utils import get_jsonl_data
from inline_coder.utils.decorators import Decorator
from inline_coder.ResultProcess.result_process import DownStreamAttach, ResultProcess
from inline_coder.Models.services import (
    GPT5_service,
    Qwen3_Coder_service,
    deepseek_service,
)
from inline_coder.utils.progress_manager import ProgressManager
from inline_coder.Parser.python_inline import PythonParser as PythonInlineParser

logger = LoggerUtils.get_main_logger(
    name=__name__,
    log_file="LOGS/benchmark/DevEval/total_logs/generate.log",
)


class Generation:
    service_func = deepseek_service

    @classmethod
    def init_generator(cls, model: ModelName):
        if model == ModelName.DeepSeekV3:
            cls.service_func = deepseek_service
        elif model == ModelName.Qwen3_Coder:
            cls.service_func = Qwen3_Coder_service
        elif model == ModelName.GPT5:
            cls.service_func = GPT5_service
        else:
            raise NotImplementedError(f"Model {model} not implemented")

    @classmethod
    @Decorator.report
    @Decorator.with_naming
    def UpDownStreamGeneration(
        cls, args: dict
    ) -> tuple[Union[str, Path], Optional[dict]]:
        data_path = args.get("data_path")
        assert data_path is not None
        limit = args.get("limit", -1)
        save_dir: Optional[Path] = args.get("save_dir")
        assert save_dir is not None, "save_dir is None"
        online = args.get("online", False)
        sample_num = args.get("sample_num", 1)
        restart = args.get("restart", False)

        dataset = get_jsonl_data(data_path)
        if limit > 0:
            dataset = dataset[:limit]
        extracted_data = []
        extracted_data = get_jsonl_data(
            PathConfig.BenchmarkDir / "extract_save" / "data.jsonl"
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

                if "target_file" not in sample:
                    target_file_path = os.path.join(
                        sample["target_project_path"],
                        "/".join(sample["completion_path"].split("/")[2:]),
                    )
                    if target_file_path.startswith("../"):
                        target_file_path = target_file_path[3:]
                    with open(target_file_path, "r", encoding="utf-8") as f:
                        sample["target_file"] = f.read()
                    # get target_signature
                    file_lines = sample["target_file"].split("\n")
                    signature_lines = file_lines[
                        sample["signature_position"][0]
                        - 1 : sample["signature_position"][1]
                    ]
                    sample["target_signature"] = "\n".join(signature_lines)

                new_data = {
                    "task_id": i,
                    "task_1_prompts": list(),
                    "task_1_predictions": list(),
                    "prompts": list(),
                    "predictions": list(),
                }
                base_prompt = DevEvalPromptBuilder.BasePrompt(sample)
                for j in tqdm(range(sample_num), desc="Batch Generate for one sample"):
                    base_response = cls.service_func(base_prompt, online)
                    new_data["task_1_prompts"].append(base_prompt)
                    new_data["task_1_predictions"].append(base_response)

                    target_signaure = sample["target_signature"]
                    result_info = ResultProcess.with_downstream_parse(
                        target_signature=target_signaure,
                        gen_prediction=base_response["response"],
                    )
                    solution_fn = result_info["target_function"]

                    # Downstream Part
                    matched_functions = []
                    for extracted_sample in extracted_data:
                        if (
                            extracted_sample["project_path"]
                            == sample["target_project_path"]
                        ):
                            matched_functions = extracted_sample["functions"]
                            if matched_functions is None:
                                matched_functions = []
                            break
                    useful_downstream = DownStreamAttach.GetUsefuleDownStreams(
                        target_function_name=sample["target_function_name"],
                        function=solution_fn,
                        ai_pred_downstream=result_info["down_stream"],
                        candidate_functions=matched_functions,
                    )
                    # PPL Part
                    full_text = base_prompt + base_response["response"]
                    ppl_result = PPLEvaluator.get_ppl(
                        text={
                            "full_text": full_text,
                            "prefix_text": base_prompt,
                        },
                        condition_mode="prefix",
                    )

                    # Upstream Part
                    inlined_result_list = list()
                    caller_info_list = []
                    for up_func in sample["upstream"]:
                        call_edge_info = {
                            "caller": up_func["content"],
                            "callee": solution_fn,
                        }
                        caller_info_list.append(call_edge_info)
                        inlined_result = PythonInlineParser.InlineMethod(call_edge_info)
                        inlined_result_list.append(inlined_result)

                    inject_prompt = DevEvalPromptBuilder.UpDownInjectPrompt(
                        sample,
                        inlined_result_list,
                        useful_downstream,
                        solution_fn,
                        ppl_result,
                    )

                    inject_response = cls.service_func(inject_prompt, online)

                    new_data["prompts"].append(inject_prompt)
                    new_data["predictions"].append(inject_response)
                progress_manager.update(advance=1, new_data=new_data)
        return save_dir, None


class Pipeline:

    @classmethod
    def ALL(cls):
        TEST_DATA_LIMIT = -1
        restart_flag = False
        model_list: list[ModelName] = [ModelName.DeepSeekV3]
        for model in model_list:
            Generation.init_generator(model)
            model_name = model.value

            generation_dir = (
                PathConfig.BenchmarkDir
                / "main_table_v1"
                / "total_gen"
                / model_name
                / "generations"
            )
            args = {
                "data_path": PathConfig.OriginDataPath,
                "limit": TEST_DATA_LIMIT,
                "save_dir": generation_dir,
                "online": True,
                "sample_num": 1,
                "restart": restart_flag,
            }
            Generation.UpDownStreamGeneration(args)
            # Evaluate Parts
            Similarity.CalculateSimilarity(
                origin_data_path=PathConfig.BenchmarkDir
                / "processed_clean_data"
                / "data.jsonl",
                generations_dir=generation_dir,
                save_dir=generation_dir.parent / "similarity",
                restart=True,
            )
            Similarity.GetAverageScore(generation_dir.parent / "similarity")


if __name__ == "__main__":
    Pipeline.ALL()
