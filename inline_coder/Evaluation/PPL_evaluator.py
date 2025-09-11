from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from rich import print
from typing import Union


class ModelLoad:
    model_name_or_path = "Qwen/Qwen2.5-Coder-1.5B"
    TOKENIZER = None
    MODEL = None
    device = None

    @classmethod
    def init(cls):
        cls.TOKENIZER = AutoTokenizer.from_pretrained(cls.model_name_or_path)
        cls.MODEL = AutoModelForCausalLM.from_pretrained(
            cls.model_name_or_path, device_map="auto", torch_dtype=torch.float16
        ).eval()
        cls.TOKENIZER.pad_token_id = cls.TOKENIZER.eos_token_id
        cls.device = cls.MODEL.device


class PPLEvaluator:
    NAME = "Perplexity"

    @staticmethod
    def evaluate_one_pair(predict):
        if ModelLoad.MODEL is None:
            ModelLoad.init()
        assert ModelLoad.MODEL is not None, "Model not initialized"
        assert ModelLoad.TOKENIZER is not None, "Tokenizer not initialized"
        assert ModelLoad.device is not None, "Device not initialized"
        inputs = ModelLoad.TOKENIZER(predict, return_tensors="pt")
        inputs = inputs.to(ModelLoad.device)
        outputs = ModelLoad.MODEL(
            **inputs,
            max_new_tokens=5,
            return_dirct_in_generate=True,
            output_scores=True,
        )
        transition_scores = ModelLoad.MODEL.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        input_length = inputs.input_ids.shape[1]
        predicted_tokens = outputs.sequences[:, :input_length]
        for tok, score in zip(predicted_tokens[0], transition_scores[0]):
            print(
                f"| {tok:5d} | {ModelLoad.TOKENIZER.decode(tok):8s} | {score.numpy():.4f} | {np.exp(score.numpy()):.2%}"
            )
        # Calculate perplexity
        ppl = 1.0
        for score in transition_scores[0]:
            ppl *= np.exp(score.numpy())
        ppl = ppl ** (1.0 / len(transition_scores[0]))
        return ppl

    @staticmethod
    def get_ppl(
        text: Union[str, dict],
        attention_mask=None,
        return_kv=False,
        end=None,
        condition_mode: str = "none",
    ):
        """
        Calculate perplexity for the given text.

        Args:
            text: The text to calculate perplexity for
            input_ids, attention_mask, past_key_values: Optional pre-processed inputs
            return_kv: Whether to return key-values
            end: End position for calculation
            condition_mode: Mode for conditional perplexity (none, prefix)

        Returns:
            A dictionary with perplexity scores and processing information
        """
        if ModelLoad.MODEL is None:
            ModelLoad.init()
        assert ModelLoad.MODEL is not None, "Model not initialized"
        assert ModelLoad.TOKENIZER is not None, "Tokenizer not initialized"
        # Use ModelLoad's tokenizer and model
        tokenizer = ModelLoad.TOKENIZER
        model = ModelLoad.MODEL
        # token length limit
        max_token_length = 25000
        if condition_mode == "prefix":
            assert isinstance(text, dict)
            full_text = text["full_text"]
            full_encoding = tokenizer(full_text, return_tensors="pt", padding=True)
            full_input_ids = full_encoding["input_ids"]
            if full_input_ids.shape[1] > max_token_length:
                # 截取后面的2500个token
                full_input_ids = full_input_ids[:, -max_token_length:]
                text["full_text"] = tokenizer.decode(
                    full_input_ids[0], skip_special_tokens=True
                )
        else:
            assert isinstance(text, str)
            full_text = text
            full_encoding = tokenizer(full_text, return_tensors="pt", padding=True)
            full_input_ids = full_encoding["input_ids"]
            if full_input_ids.shape[1] > max_token_length:
                # 截取后面的2500个token
                full_input_ids = full_input_ids[:, -max_token_length:]
                text = tokenizer.decode(full_input_ids[0], skip_special_tokens=True)

        # Initialize input processing
        condition_pos_id = 0
        if condition_mode == "prefix":
            assert isinstance(text, dict)
            assert "full_text" in text
            assert "prefix_text" in text
            encoding = tokenizer(text["full_text"], return_tensors="pt", padding=True)
            input_ids = encoding["input_ids"].to(model.device)  # type: ignore
            attention_mask = encoding["attention_mask"].to(model.device)  # type: ignore
            prefix_text = text["prefix_text"]
            prefix_token_count = len(
                ModelLoad.TOKENIZER(prefix_text, return_tensors="pt")["input_ids"][0]  # type: ignore
            )
            condition_pos_id = prefix_token_count
        else:
            assert isinstance(text, str)
            encoding = tokenizer(text, return_tensors="pt", padding=True)
            input_ids = encoding["input_ids"].to(model.device)  # type: ignore
            attention_mask = encoding["attention_mask"].to(model.device)  # type: ignore

        if end is None:
            end = input_ids.shape[1]
        end = min(end, model.config.max_position_embeddings)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids[:, :end],
                attention_mask=attention_mask[:, :end],  # type: ignore
                return_dict=True,
                output_hidden_states=True,
                use_cache=True,
            )

        # Get logits and shift
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:end].contiguous()

        # Flatten tokens for loss calculation
        active = (attention_mask[:, :end] == 1)[..., :-1].view(-1)  # type: ignore
        active_logits = shift_logits.view(-1, shift_logits.size(-1))[active]
        active_labels = shift_labels.view(-1)[active]

        # Calculate loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(active_logits, active_labels)

        # Apply condition filtering if required
        if condition_mode == "prefix":
            loss = loss[condition_pos_id:]

        # Calculate mean perplexity
        mean_loss = loss.mean() if len(loss) > 0 else torch.tensor(0.0)
        ppl = (
            torch.exp(mean_loss).item()
            if mean_loss.item() != float("inf")
            else float("inf")
        )

        result = {
            "loss": loss,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "ppl": ppl,
        }

        if return_kv:
            result["past_key_values"] = outputs.past_key_values

        return result


# Expected output:
if __name__ == "__main__":
    # Example usage
    example_text = "This is an example text to evaluate."
    maybe_input_list = [
        "hello world",
        "but this is a test",
        "123",
    ]
    predict = "456"
    evaluator = PPLEvaluator()
    for inp in maybe_input_list:
        full_text = inp + predict
        input_token_count = len(
            ModelLoad.TOKENIZER(inp, return_tensors="pt")["input_ids"][0]  # type: ignore
        )
        result = evaluator.get_ppl(
            text={
                "full_text": full_text,
                "prefix_text": inp,
            },
            condition_mode="prefix",
        ).get("ppl", None)
        print(f"full text: [green]{inp}[/]->[red]{predict}[/], ppl: {result}")
    print("*****************")
    for inp in maybe_input_list:
        full_text = inp + predict
        input_token_count = len(
            ModelLoad.TOKENIZER(inp, return_tensors="pt")["input_ids"][0]  # type: ignore
        )
        result = evaluator.get_ppl(text=full_text).get("ppl", None)
        print(f"full text: [green]{inp}[/]->[red]{predict}[/], ppl: {result}")
