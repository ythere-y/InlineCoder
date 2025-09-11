from functools import singledispatch
import logging
from types import TracebackType
from typing import Iterable, Optional, Type
import pickle
import jsonlines
import json
import traceback
import time
from typing import Dict, Any, Union, Callable
from pathlib import Path
from rich.text import Text
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    SpinnerColumn,
    TimeElapsedColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    ProgressColumn,
)
from .context_manager import ensure_directory_exists

logger = logging.getLogger(__name__)


class TransferSpeedColumn(ProgressColumn):
    """Renders human readable transfer speed."""

    def render(self, task) -> Text:
        """Show data process speed."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("?", style="progress.data.speed")
        data_speed = int(speed)
        return Text(f"{data_speed}iter/s", style="progress.data.speed")


class ProgressManager:

    def __init__(
        self,
        task_name: str,
        data_num: int,
        save_dir: Union[str, Path],
        on_error: Optional[Callable[..., Any]] = None,
        restart: bool = False,
        remove_existing: bool = False,
        save_as_pkl: bool = False,
    ):
        self.save_dir = Path(save_dir)
        self.save_as_pkl = save_as_pkl
        if self.save_as_pkl:
            self.data_path = self.save_dir / "data.pkl"
        else:
            self.data_path = self.save_dir / "data.jsonl"
        self.info_path = self.save_dir / "info.json"
        self.data_num = data_num
        self.info = {
            "task_name": task_name,
            "save_dir": str(self.save_dir),
            "start_date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "end_date": None,
            "total_data": 0,
            "processed_data_num": 0,
            "time_spent": 0,
            "error": None,
        }
        self.processed_data = []
        self.task_name = task_name
        self.progress_bar = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TransferSpeedColumn(),
            MofNCompleteColumn(),
        )
        self.columns_in_log = [
            TextColumn("[progress.description]{task.description}"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TransferSpeedColumn(),
        ]
        self.on_error = on_error
        self.restart = restart
        self.remove_existing = remove_existing
        self.check_data_path()

    def check_data_path(self):
        with ensure_directory_exists(self.save_dir):
            if self.restart:
                # check if the data_path exists
                for path in [self.data_path, self.info_path]:
                    if path.exists():
                        if self.remove_existing:
                            logger.info(
                                f"[bold red]file {path} already exists, will recover it...[/]"
                            )
                        else:
                            logger.warning(
                                f"[bold red]Data file {path} already exists, "
                                "please delete it before running the program[/]"
                            )
                            # end the program
                            raise FileExistsError(
                                f"[bold red]Data file {path} already exists, "
                                "please delete it before running the program[/]"
                            )
            else:
                pass

    def __enter__(self):
        self.progress_bar.__enter__()
        self.task = self.progress_bar.add_task(self.task_name, total=self.data_num)
        if not self.restart:
            # load progress
            if self.data_path.exists():
                if self.save_as_pkl:
                    with open(self.data_path, "rb") as f:
                        self.processed_data = pickle.load(f)
                else:
                    with jsonlines.open(self.data_path, "r") as reader:
                        self.processed_data = list(reader)
                self.progress_bar.update(
                    self.task,
                    completed=len(self.processed_data),
                    description=f"Processing {self.task_name}",
                )

                logger.info(f"[bold green]Loaded data from {self.data_path}[/]")
                logger.info(f"[bold green]Restart from {len(self.processed_data)}[/]")

            if self.info_path.exists():
                with open(self.info_path, "r") as f:
                    saved_info = json.load(f)
                    self.info["processed_data_num"] = saved_info["processed_data_num"]
                    logger.info(f"[bold blue]Loaded info from {self.info_path}[/]")

        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        logger.debug("[bold blue]Exiting progress manager[/]")
        self.progress_bar.__exit__(exc_type, exc_val, exc_tb)
        self.info["end_date"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.info["time_spent"] = time.time() - time.mktime(
            time.strptime(self.info["start_date"], "%Y-%m-%d %H:%M:%S")
        )
        self.info["total_data"] = self.data_num
        self.info["processed_data_num"] = len(self.processed_data)
        self.info["error"] = None
        self.error = None

        # error handling
        if exc_val is not None:
            self.error = {
                "type": exc_type.__name__ if exc_type else None,
                "message": str(exc_val),
                "traceback": "".join(
                    traceback.format_exception(exc_type, exc_val, exc_tb)
                ),
            }
            self.info["error"] = self.error
            logger.error(f"[bold red]Error: {self.error['message']}[/]")
            if self.on_error is not None:
                self.on_error(exc_val)

        self.save_progress()
        if exc_type is not None:
            logger.error(exc_val)
        return

    def log_progress(self):
        """Log the progress of the task."""
        task = self.progress_bar.tasks[self.task]
        text_list = [column.render(task) for column in self.columns_in_log]
        text_line = Text(" ").join(text_list)
        logger.debug(text_line)

    def update(
        self,
        advance: int = 1,
        new_data: Optional[dict] = None,
    ):
        self.progress_bar.update(
            self.task,
            advance=advance,
            description=f"Processing {self.task_name}",
        )
        self.processed_data.append(new_data)
        self.log_progress()

    def save_progress(self):
        """Save the progress to a file."""
        logger.debug(f"[bold blue]Saving progress to dir {self.save_dir}[/]")
        # use jsonlines to save the data
        if self.save_as_pkl:
            with open(self.data_path, "wb") as f:
                logger.info(f"[bold blue]Saving data to {self.data_path}[/]")
                pickle.dump(self.processed_data, f)
        else:
            with jsonlines.open(self.data_path, "w") as writer:
                logger.info(f"[bold blue]Saving data to {self.data_path}[/]")
                for data in self.processed_data:
                    writer.write(data)
        # save the information to a file
        with open(self.info_path, "w") as f:
            logger.info(f"[bold blue]Saving info to {self.info_path}[/]")
            json.dump(self.info, f, indent=4, ensure_ascii=False)


def test_progress_manager():
    data_dir = Path("../data/Approach4/generated_data/DeepSeek-V3/")
    data_path = data_dir / "0-80-processed.jsonl"
    info_path = data_dir / "0-80-info.json"
    source_data = {
        "data": data_path,
        "info": info_path,
    }
    total_data = load_data(source_data)
    assert isinstance(total_data, Dict)
    assert isinstance(total_data.get("data"), Iterable)
    data = total_data.get("data")
    assert isinstance(data, Iterable)
    data_len = 0
    with jsonlines.open(data_path, "r") as reader:
        data_len = sum(1 for _ in reader)
    with ProgressManager(
        task_name="Test Progress Manager",
        data_num=data_len,
        save_dir="../data/test/DeepSeek-V3/",
    ) as progress_manager:
        for idx, data in enumerate(data):
            if idx < len(progress_manager.processed_data):
                # skip the data that has been processed
                continue
            time.sleep(0.06)
            if idx == 20:
                bar = 0 / 0
            new_data = {
                "processed_data": {
                    "id": idx,
                },
                "prcessed": True,
            }
            progress_manager.update(new_data=new_data)


@singledispatch
def load_data(
    data_source,
) -> Union[Dict[str, Any], Iterable[Union[Dict[str, Any], str]]]:
    """Load data from the data source."""
    raise NotImplementedError("Unsupported data source type")


@load_data.register(Path)
@load_data.register(str)
def _(data_source: Union[Path, str]) -> Iterable[Union[Dict[str, Any], str]]:
    if isinstance(data_source, str):
        data_source = Path(data_source)
    if data_source.suffix == ".jsonl":
        with jsonlines.open(data_source) as reader:
            for obj in reader:
                yield obj
    elif data_source.suffix == ".json":
        with open(data_source, "r") as f:
            data = json.load(f)
            for obj in data:
                yield obj
    else:
        # read as text file,read all and return all
        with open(data_source, "r") as f:
            data = f.read()
            yield data


@load_data.register(dict)
def _(data_source: dict) -> Dict[str, Any]:
    data = {}
    for key, value in data_source.items():
        data[key] = load_data(value)
    return data


if __name__ == "__main__":
    test_progress_manager()
