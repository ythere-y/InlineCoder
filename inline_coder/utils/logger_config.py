import logging
from rich.logging import RichHandler


class LoggerUtils:
    @staticmethod
    def logging_ignore_warnings():
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("httpcore.connection").setLevel(logging.WARNING)
        logging.getLogger("httpcore.http11").setLevel(logging.WARNING)
        logging.getLogger("openai._base_client").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("codetext.parser.python_parser").setLevel(logging.WARNING)

    @staticmethod
    def get_main_logger(
        name: str,
        log_file: str,
        file_log_level=logging.DEBUG,
        stream_log_level=logging.INFO,
    ) -> logging.Logger:
        LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        DATE_FORMAT = "[%Y-%m-%d %H:%M:%S]"

        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(file_log_level)
        logging.basicConfig(
            level="DEBUG",
            format=LOG_FORMAT,
            datefmt=DATE_FORMAT,
            handlers=[
                RichHandler(rich_tracebacks=True, markup=True, level=stream_log_level),
                file_handler,
            ],
        )

        logger = logging.getLogger(name)

        return logger


LoggerUtils.logging_ignore_warnings()
