import time
import threading


def timeout_tracker(timeout_duration, return_value=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = [None]

            file_info = args[0]

            def target():
                result[0] = func(*args, **kwargs)

            thread = threading.Thread(target=target)
            thread.start()
            thread.join(timeout_duration)

            if thread.is_alive():
                print(
                    f"function: '{func.__name__}' operation target {file_info['file_path']} with star = {file_info['star']} overtime, return {return_value}"
                )
                return return_value
            else:
                return result[0]

        return wrapper

    return decorator
