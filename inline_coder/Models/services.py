from openai import OpenAI
import time
import logging
from inline_coder.utils.decorators import Decorator
from inline_coder.utils.data_utils import get_json_data

logger = logging.getLogger(__name__)
configs = get_json_data("configs/CONFIGS.json", logger)
api_key = configs["api_key"]

client = OpenAI(
    base_url=configs["base_url"],
    api_key=configs["api_key"],
)


@Decorator.retry(times=3, exceptions=(Exception,))
def get_deepseekv3_response(content) -> str:
    """
    Get response from DeepSeek-V3 model.

    Args:
        content (str): The input content for the model.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling parameter.

    Returns:
        str: The response from the model.
    """
    response = client.chat.completions.create(
        model="DeepSeek-V3",
        messages=[{"role": "user", "content": content}],
        temperature=0.00,
    )
    content = response.choices[0].message.content
    if content is None:
        content = "No response from DeepSeek-V3."
    return content


@Decorator.retry(times=3, exceptions=(Exception,))
def get_Qwen3_Coder_response(content) -> str:
    """
    Get response from Qwen3-Coder model.

    Args:
        content (str): The input content for the model.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling parameter.

    Returns:
        str: The response from the model.
    """
    response = client.chat.completions.create(
        model="Qwen3-Coder",
        messages=[{"role": "user", "content": content}],
        temperature=0.00,
    )
    content = response.choices[0].message.content
    if content is None:
        content = "No response from Qwen3-Coder."
    return content


@Decorator.retry(times=3, exceptions=(Exception,))
def get_GPT5_response(content) -> str:
    """
    Get response from Qwen3-Coder model.

    Args:
        content (str): The input content for the model.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling parameter.

    Returns:
        str: The response from the model.
    """

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": content}],
        temperature=0.00,
    )
    content = response.choices[0].message.content
    if content is None:
        content = "No response from GPT5."
    return content


@Decorator.retry(times=3, exceptions=(Exception,))
def get_model_response(model_name: str, content) -> str:
    """
    Get response from Qwen3-Coder model.

    Args:
        content (str): The input content for the model.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling parameter.

    Returns:
        str: The response from the model.
    """
    response = client.chat.completions.create(
        model=model_name.lower(),
        messages=[{"role": "user", "content": content}],
        temperature=0.00,
    )
    content = response.choices[0].message.content
    if content is None:
        content = "No response from GPT5."
    return content


def deepseek_service(content, online: bool = False) -> dict:
    result = {
        "time_cost": 0,
        "response": "",
    }
    if not online:
        result = {
            "time_cost": 0,
            "response": "DeepSeek-V3 is not available in offline mode.",
        }
    else:
        start_time = time.time()
        try:
            result["response"] = get_deepseekv3_response(content)
        except Exception as e:
            result["response"] = str(e)
        end_time = time.time()
        result["time_cost"] = end_time - start_time
    if "Error code: 503" in result["response"]:
        logger.warning(f"Error code 503 encountered!")
    return result


def Qwen3_Coder_service(content, online: bool = False) -> dict:
    result = {
        "time_cost": 0,
        "response": "",
    }
    if not online:
        result = {
            "time_cost": 0,
            "response": "Qwen3-Coder is not available in offline mode.",
        }
    else:
        start_time = time.time()
        try:
            result["response"] = get_Qwen3_Coder_response(content)
        except Exception as e:
            result["response"] = str(e)
        end_time = time.time()
        result["time_cost"] = end_time - start_time
    if "Error code: 503" in result["response"]:
        logger.warning(f"Error code 503 encountered!")
    return result


def GPT5_service(content, online: bool = False) -> dict:
    result = {
        "time_cost": 0,
        "response": "",
    }
    if not online:
        result = {
            "time_cost": 0,
            "response": "GPT5 is not available in offline mode.",
        }
    else:
        start_time = time.time()
        try:
            result["response"] = get_GPT5_response(content)
        except Exception as e:
            result["response"] = str(e)
        end_time = time.time()
        result["time_cost"] = end_time - start_time
    if "Error code: 503" in result["response"]:
        logger.warning(f"Error code 503 encountered!")
    return result


def model_service(
    model_name: str,
    content: str,
    online: bool = False,
) -> dict:
    result = {
        "time_cost": 0,
        "response": "",
    }
    if not online:
        result = {
            "time_cost": 0,
            "response": f"{model_name} is not available in offline mode.",
        }
    else:
        start_time = time.time()
        try:
            result["response"] = get_model_response(model_name, content)
        except Exception as e:
            result["response"] = str(e)
        end_time = time.time()
        result["time_cost"] = end_time - start_time
    if "Error code: 503" in result["response"]:
        logger.warning(f"Error code 503 encountered!")
    return result


def test_service():
    content = "What is the capital of France?"
    result = deepseek_service(content, online=True)
    print(f"Response: {result['response']}")
    print(f"Time cost: {result['time_cost']} seconds")


def test_response():
    content = "What is the capital of France?"
    # response = get_deepseekv3_response(content)
    # response = get_Qwen3_Coder_response(content)
    response = GPT5_service(content, online=True)
    print(f"Response: {response}")
    response = deepseek_service(content, online=True)
    print(f"Response: {response}")
    response = Qwen3_Coder_service(content, online=True)
    print(f"Response: {response}")


if __name__ == "__main__":
    test_response()
