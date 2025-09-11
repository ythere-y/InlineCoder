import time
from functools import wraps
import logging


class Decorator:
    logger = logging.getLogger("Decorator")

    @classmethod
    def retry(cls, times, exceptions):
        """
        Retry Decorator
        Retries the wrapped function/method `times` times if the exceptions listed
        in ``exceptions`` are thrown
        :param times: The number of times to repeat the wrapped function/method
        :type times: Int
        :param Exceptions: Lists of exceptions that trigger a retry attempt
        :type Exceptions: Tuple of Exceptions
        """

        def decorator(func):
            def newfn(*args, **kwargs):
                attempt = 0
                while attempt < times:
                    try:
                        return func(*args, **kwargs)
                    except exceptions:
                        cls.logger.warning(
                            "Exception thrown when attempting to run %s, attempt "
                            "%d of %d" % (func, attempt, times)
                        )
                        attempt += 1
                return func(*args, **kwargs)

            return newfn

        return decorator

    @classmethod
    def timing(cls, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            cls.logger.info(
                f"Function '{func.__name__}' executed in {end_time - start_time:.4f}s"
            )
            return result

        return wrapper

    @classmethod
    def with_naming(cls, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cls.logger.info(
                f"[bold green]Running function '{func.__name__}'[/bold green]"
            )
            # 打印函数参数
            if args:
                cls.logger.info(f"Arguments: {args}")
            if kwargs:
                cls.logger.info(f"Keyword arguments: {kwargs}")
            result = func(*args, **kwargs)
            cls.logger.info(
                f"[bold blue]Finished function '{func.__name__}'[/bold blue]"
            )
            return result

        return wrapper

    @classmethod
    def report(cls, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            time_usage = time.time() - start_time
            report_messages = {
                "function_name": func.__name__,
                "args": args,
                "kwargs": kwargs,
                "time_usage": time_usage,
                "function_result": result,
            }
            cls.logger.info("[bold red] Report result:\n[/]")
            cls.logger.info(report_messages)
            return result

        return wrapper


if __name__ == "__main__":

    @Decorator.retry(times=3, exceptions=(ValueError,))
    def test_():
        print("Running test function")
        raise ValueError("This is a test exception")

    try:
        test_()
    except ValueError as e:
        print(f"Caught an exception: {e}")
