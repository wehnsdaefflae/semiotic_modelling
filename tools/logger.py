# coding=utf-8
import datetime
import os
import sys
from typing import Sequence, Optional, Dict, Tuple


class Logger:
    _time = datetime.datetime.now()
    _file_path = sys.argv[0]
    _base_name = os.path.basename(_file_path)
    _first_name = os.path.splitext(_base_name)[0]
    _time_str = _time.strftime("%Y-%m-%d_%H-%M-%S")
    _main_name = _first_name + _time_str

    @staticmethod
    def log(content: str):
        Logger._log(Logger._main_name + ".log", content)

    @staticmethod
    def _log(target: str, content: str):
        print(content)
        now = datetime.datetime.now()
        now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        with open(target, mode="a") as file:
            file.write(now_str + "\t" + content + "\n")


class DataLogger:
    @staticmethod
    def log_to(file_path: str, header: Sequence[str], data: Sequence[str]):
        if not len(header) == len(data):
            raise ValueError("inconsistent sizes")

        content = "\t".join(data)
        if not os.path.isfile(file_path):
            content = "\t".join(header) + "\n" + content

        with open(file_path, mode="a") as file:
            file.write(content + "\n")
