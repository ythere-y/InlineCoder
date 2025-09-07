import sys
import os
from typing import Any
from Evaluation.EM_evaluator import EMEvaluator
from Evaluation.ES_evaluator import ESEvaluator
from Evaluation.PPL_evaluator import PPLEvaluator

from Evaluation.BLEU_evaluator import BLEUEvaluator


class MetricProcess:
    @classmethod
    def process(cls, parsed_solution: dict, args: dict) -> Any:
        """process calculate the score and return the results in a list.

        Args:
            parsed_solution (dict):
            {
                "reference":str,
                "generations":list[list[dict]],
            }
            one_generation_dict = {
                "prompt":str,
                "resposne":{
                    "target_function":str,
                    "target_signature":str,
                    "target_function_body":str,
                    "down_stream":str,
                }
            }
            args (dict):

        Returns:
            list: list[list[float]]
        """
        raise NotImplementedError("Subclasses should implement this method.")

    NAME: str = "None"


class BLEUProcess(MetricProcess):
    NAME = BLEUEvaluator.NAME

    @classmethod
    def process(cls, parsed_solution: dict, args: dict) -> list:
        reference = parsed_solution["reference"]
        score_lists = []
        for gen_list in parsed_solution["generations"]:
            score_list = []
            for gen in gen_list:
                score = BLEUEvaluator.evaluate_one_pair(
                    predict=gen["response"]["target_function"],
                    reference=reference,
                )
                score_list.append(score)
            score_lists.append(score_list)
        return score_lists


class EMProcess(MetricProcess):
    NAME = EMEvaluator.NAME

    @classmethod
    def process(cls, parsed_solution: dict, args: dict) -> list:
        reference = parsed_solution["reference"]
        score_lists = []
        for gen_list in parsed_solution["generations"]:
            score_list = []
            for gen in gen_list:
                score = EMEvaluator.evaluate_one_pair(
                    predict=gen["response"]["target_function"],
                    reference=reference,
                )
                score_list.append(score)
            score_lists.append(score_list)
        return score_lists


class ESProcess(MetricProcess):
    NAME = ESEvaluator.NAME

    @classmethod
    def process(cls, parsed_solution: dict, args: dict) -> list:
        reference = parsed_solution["reference"]
        score_lists = []
        for gen_list in parsed_solution["generations"]:
            score_list = []
            for gen in gen_list:
                score = ESEvaluator.evaluate_one_pair(
                    predict=gen["response"]["target_function"],
                    reference=reference,
                )
                score_list.append(score)
            score_lists.append(score_list)
        return score_lists


class PPLProcess(MetricProcess):
    NAME = PPLEvaluator.NAME

    @classmethod
    def process(cls, parsed_solution: dict, args: dict) -> list:
        score_lists = []
        for gen_list in parsed_solution["generations"]:
            score_list = []
            for gen in gen_list:
                full_text = gen["prompt"] + "\n" + gen["response"]["target_function"]

                score = PPLEvaluator.get_ppl(
                    text={
                        "full_text": full_text,
                        "prefix_text": gen["prompt"],
                    },
                    condition_mode="prefix",
                ).get("ppl", 0)
                score_list.append(score)
            score_lists.append(score_list)
        return score_lists
