# coding=utf-8
import time


class Timer:
    _last_time = -1  # type: int

    @staticmethod
    def time_passed(passed_time_ms: int) -> bool:
        if 0 >= passed_time_ms:
            raise ValueError("Only positive millisecond values allowed.")

        this_time = round(time.time() * 1000.)

        if Timer._last_time < 0:
            Timer._last_time = this_time
            return False

        elif this_time - Timer._last_time < passed_time_ms:
            return False

        Timer._last_time = this_time
        return True
