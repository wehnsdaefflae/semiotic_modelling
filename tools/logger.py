# coding=utf-8
import datetime
import os
import sys
from typing import Sequence, Optional


def get_time_string():
    _time = datetime.datetime.now()
    return _time.strftime("%Y-%m-%d_%H-%M-%S")


def get_main_script_name():
    _file_path = sys.argv[0]
    _base_name = os.path.basename(_file_path)
    return os.path.splitext(_base_name)[0]


class Logger:
    _file_name = get_main_script_name() + "_" + get_time_string() + ".log"
    dir_path = "logs/"

    @staticmethod
    def log(content: str):
        assert Logger.dir_path.endswith("/")
        if not os.path.isdir(Logger.dir_path):
            os.makedirs(Logger.dir_path)
        Logger._log(Logger.dir_path + Logger._file_name, content)

    @staticmethod
    def _log(file_path: str, content: str):
        print(content)
        now = datetime.datetime.now()
        now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        with open(file_path, mode="a") as file:
            file.write(now_str + "\t" + content + "\n")


class DataLogger:
    _main_name = get_main_script_name() + "_" + get_time_string() + ".log"

    @staticmethod
    def log_to(header: Sequence[str], data: Sequence[str], file_name: Optional[str] = None, dir_path: Optional[str] = None):
        if not len(header) == len(data):
            raise ValueError("inconsistent sizes")

        if file_name is None:
            file_name = DataLogger._main_name

        if dir_path is None:
            file_path = file_name

        else:
            assert dir_path.endswith("/")
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)
                Logger.log(f"{DataLogger.__class__.__name__:s} directory '{dir_path:s}' created.")

            file_path = dir_path + file_name

        content = "\t".join(data)
        if not os.path.isfile(file_path):
            content = "\t".join(header) + "\n" + content

        with open(file_path, mode="a") as file:
            file.write(content + "\n")
