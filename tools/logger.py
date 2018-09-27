# coding=utf-8
import datetime
import os
import sys


class Logger:
    _time = datetime.datetime.now()
    _file_path = sys.argv[0]
    _base_name = os.path.basename(_file_path)
    _first_name = os.path.splitext(_base_name)[0]
    _time_str = _time.strftime("%Y-%m-%d_%H-%M-%S")
    log_name = _first_name + _time_str + ".log"

    @staticmethod
    def log(message: str):
        print(message)
        now = datetime.datetime.now()
        now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        with open(Logger.log_name, mode="a") as file:
            file.write(now_str + "\t" + message + "\n")