# coding=utf-8
import datetime
import os
import sys
from typing import Sequence, Any, Optional


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

    @staticmethod
    def log_to(target: str, *data: str, header: Optional[Sequence[str]] = None):
        row = "\t".join(data)
        file_name = Logger._main_name + "_" + target + ".log"
        if header is not None:
            if not len(data) == len(header):
                raise ValueError("Data and header have different lengths.")
            if not os.path.isfile(file_name):
                with open(file_name, mode="a") as file:
                    header_row = "\t".join(header)
                    file.write(header_row + "\n")
        Logger._log(file_name, row)
