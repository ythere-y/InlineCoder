import os
import json
import difflib
from colorama import Fore, Style
import logging
from logging import Logger
from typing import Optional

logger = logging.getLogger(__name__)


def get_clean_code(code):
    code = code.strip()
    if code.startswith("```java"):
        code = code.replace("```java", "")
    if code.endswith("```"):
        code = code.replace("```", "")
    code = code.strip()
    return code


def string_diff(
    str_org,
    str_new,
    title: str = "diff",
    logger: Optional[Logger] = logger,
    display=True,
    display_org=False,
    display_new=False,
):
    diff = difflib.ndiff(str_org.splitlines(), str_new.splitlines())
    if display_org and display_new:
        from rich import print
        from rich.panel import Panel
        from rich.columns import Columns
        from rich.syntax import Syntax

        str_org = "\n".join(str_org.splitlines())
        str_new = "\n".join(str_new.splitlines())
        syntax_org = Syntax(str_org, "python", theme="monokai", line_numbers=True)
        syntax_new = Syntax(str_new, "python", theme="monokai", line_numbers=True)
        title = f"{title} - {len(str_org.splitlines())} lines"
        panel_org = Panel(syntax_org, title=f"{title} - [bold]Original[/]", expand=True)
        panel_new = Panel(syntax_new, title=f"{title} - [bold]New[/]", expand=True)
        print(Columns([panel_org, panel_new]))

    if display:
        if logger is not None:
            logger.debug(f"[bold red]{title}:[/]")
            for idx, line in enumerate(diff):
                if line.startswith("+ "):
                    logger.debug(f"[green][{idx}] {line}[/]")
                elif line.startswith("- "):
                    logger.debug(f"[red][{idx}] {line}[/]")
                else:
                    logger.debug(f"[{idx}] {line}")
        else:
            from rich import print

            print(f"[bold cyan]{title}:[/]")
            for line in diff:
                if line.startswith("+ "):
                    print(f"[green]{line}[/]")
                elif line.startswith("- "):
                    print(f"[red]{line}[/]")
                else:
                    print(line)

    return diff


def make_dir(path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))


def get_ranged_jsonl_data(data_path, range=None, logger=logger):
    data = get_jsonl_data(data_path, logger)
    if range is None:
        return data
    elif range[1] == -1:
        return data
    else:
        return data[range[0] : range[1]]


def backup_data(data_path, logger=logger):
    backup_path = data_path.replace("/data/", "/tmp/")
    if logger is not None:
        logger.info(f"backup data to {backup_path}")
    else:
        print(f"{Style.BRIGHT}{Fore.BLUE}backup data to {backup_path}{Style.RESET_ALL}")
    make_dir(backup_path)
    with open(backup_path, "w", encoding="utf-8") as f:
        with open(data_path, "r", encoding="utf-8") as source_code:
            source_str = source_code.read()
            f.write(source_str)
    return backup_path


def save_json_data(data, file, logger=logger):
    make_dir(file)
    if logger is not None:
        logger.info(f"save data to {file}")
    else:
        print(f"{Style.BRIGHT}{Fore.BLUE}save data to {file}{Style.RESET_ALL}")
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def save_text_data(data, file, logger=logger):
    make_dir(file)
    if logger is not None:
        logger.info(f"save text to {file}")
    else:
        print(f"{Style.BRIGHT}{Fore.BLUE}save text to {file}{Style.RESET_ALL}")
    with open(file, "w", encoding="utf-8") as f:
        f.write(data)


def save_jsonl_data(data, file, logger: Optional[Logger] = logger):
    make_dir(file)
    if logger is not None:
        logger.info(f"save data to {file}")
    else:
        print(f"{Style.BRIGHT}{Fore.BLUE}save data to {file}{Style.RESET_ALL}")
    with open(file, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def get_json_data(file, logger=logger):
    data = {}
    if logger is not None:
        logger.info(f"load data from {file}")
    else:
        print(f"{Style.BRIGHT}{Fore.GREEN}load data from {file}{Style.RESET_ALL}")
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def display_data_sample(data, logger=logger):
    sample = data[0]
    if logger is not None:
        logger.info("data sample:")
        logger.info(f"Keys => {sample.keys()}")
        logger.info(f"Sample =>\n{json.dumps(sample, ensure_ascii=False, indent=4)}")
    else:
        print(f"{Style.BRIGHT}{Fore.GREEN}data sample:{Style.RESET_ALL}")
        print(f"Keys => {sample.keys()}")
        print(f"Sample =>\n{json.dumps(sample, ensure_ascii=False, indent=4)}")


def get_example(file, logger=logger):
    data = []
    if logger is not None:
        logger.info(f"load data from {file}")
    else:
        print(f"{Style.BRIGHT}{Fore.GREEN}load data from {file}{Style.RESET_ALL}")
    with open(file, "r", encoding="utf-8") as f:
        line = f.readline()
        data.append(json.loads(line))
    save_json_data(data, "LOGS/example.json", logger)
    if logger is not None:
        logger.info("save example to LOGS/example.json")
    else:
        print(
            f"{Style.BRIGHT}{Fore.BLUE}save example to LOGS/example.json{Style.RESET_ALL}"
        )


def get_jsonl_data(file, logger=logger):
    data = []
    if logger is not None:
        logger.info(f"load data from {file}")
    else:
        print(f"{Style.BRIGHT}{Fore.GREEN}load data from {file}{Style.RESET_ALL}")
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def index_jumper(data, index_list):
    # 在data中，跳过index_list中的index
    result = []
    for i in range(len(data)):
        if i in index_list:
            continue
        result.append(data[i])
    return result


def get_sample_files(repo_info_list, extension, sample_num):
    # if sample_num == -1: return all files
    result = []
    for repo_info in repo_info_list:
        direction = repo_info["file_path"]
        current_repo_file_list = []
        current_repo_results = []
        for root, dirs, files in os.walk(direction):
            for file in files:
                if file.endswith(extension):
                    if file.endswith("Tests.java") or file.endswith("Test.java"):
                        continue
                    current_file_path = os.path.join(root, file)
                    file_path_in_repo = current_file_path.split("repos/")[-1].split(
                        "/", 1
                    )[1]
                    current_repo_file_list.append(
                        {
                            "file_path": current_file_path,
                            "path_in_repo": file_path_in_repo,
                        }
                    )
                    current_repo_results.append(
                        {
                            "file_path": current_file_path,
                            "file_path_in_repo": file_path_in_repo,
                            "full_name": repo_info["full_name"],
                            "repo_name": repo_info["repo_name"],
                            "star": repo_info["star"],
                        }
                    )
        for cur_result in current_repo_results:
            cur_result["file_list"] = current_repo_file_list
        result.extend(current_repo_results)
        if len(result) > sample_num:
            break
    return result


def yield_walk(direction, extension):
    for root, dirs, files in os.walk(direction):
        for file in files:
            if file.endswith(extension):
                yield os.path.join(root, file)


def walk(direction, extension, logger=logger):
    print(
        f"{Style.BRIGHT}{Fore.GREEN}walking {direction} with extension {extension}{Style.RESET_ALL}"
    )
    result = []
    for root, dirs, files in os.walk(direction):
        for file in files:
            if file.endswith(extension):
                result.append(os.path.join(root, file))
    if logger is not None:
        logger.info(f"found {len(result)} files")
    else:
        print(f"{Style.BRIGHT}{Fore.GREEN}found {len(result)} files{Style.RESET_ALL}")
    return result


def read_file_with_any(file):
    encoding_flag = ""
    try:
        # try utf-8
        with open(file, "r", encoding="utf-8") as source_code:
            source_str = source_code.read()
            source_code.close()
            encoding_flag = "utf-8"
    except:
        # try gbk
        try:
            with open(file, "r", encoding="gbk") as source_code:
                source_str = source_code.read()
                source_code.close()
                encoding_flag = "gbk"
        except:
            # try ascii
            try:
                with open(file, "r", encoding="ascii") as source_code:
                    source_str = source_code.read()
                    source_code.close()
                    encoding_flag = "ascii"
            except:
                # try latin-1
                try:
                    with open(file, "r", encoding="latin-1") as source_code:
                        source_str = source_code.read()
                        source_code.close()
                        encoding_flag = "latin-1"
                except:
                    raise Exception(f"file encoding error, filename = {file}")
    return source_str, encoding_flag
