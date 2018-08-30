import random
import string
from math import sin, cos
from typing import Generator, Union, Tuple, Optional

from dateutil import parser
from matplotlib import pyplot

from data.data_processing import equisample, series_generator


def env_text(file_path: str) -> Generator[str, None, None]:
    permissible_non_letter = string.digits + string.punctuation + " "
    while True:
        with open(file_path, mode="r") as file:
            for line in file:
                for character in line:
                    if character in string.ascii_letters:
                        yield character.lower()

                    elif character in permissible_non_letter:
                        yield character


def _convert_to_timestamp(time_val: Optional[Union[int, str]]) -> int:
    if time_val is None:
        return -1
    time_type = type(time_val)
    if time_type == int:
        return time_val
    elif time_type == str:
        date_time = parser.parse(time_val)
        return date_time.timestamp()
    raise ValueError()


def env_crypto(file_path: str, interval_seconds: int,
               start_val: Optional[Union[int, str]] = None, end_val: Optional[Union[int, str]]=None) -> Generator[float, None, None]:
    start_ts = _convert_to_timestamp(start_val)
    end_ts = _convert_to_timestamp(end_val)

    while True:
        raw_generator = series_generator(file_path, start_timestamp=start_ts, end_timestamp=end_ts)
        for t, value in equisample(raw_generator, interval_seconds):
            yield value


def test_env_text():
    text_path = "D:/Data/Texts/pride_prejudice.txt"
    for v in env_text(text_path):
        print(v, end="")


def test_env_crypto():
    rate_path = "D:/Data/binance/01Jan2010--1m/EOSETH.csv"
    delta = 60

    start_time = 1501113720
    end_time = 1532534580

    time_axis = []
    value_axis = []

    for v in env_crypto(rate_path, start_time, end_time, delta):
        time_axis.append(start_time)
        start_time += delta
        value_axis.append(v)

    pyplot.plot(time_axis, value_axis)
    pyplot.show()


def env_ascending_descending_nominal() -> Generator[str, None, None]:
    i = 0
    length = len(string.ascii_lowercase)
    forward = True
    while True:
        yield string.ascii_lowercase[i]
        if random.random() < .1:
            forward = not forward
        i = (i + int(forward) * 2 - 1) % length


def env_trigonometric_rational() -> Generator[Tuple[float, float], None, None]:
    i = 0
    while True:
        # examples = [(sin(t / 100.), cos(t / 70.)*3. + sin(t/13.)*.7)]
        yield sin(i / 100.), float(cos(i / 100.) >= 0.) * 2. - 1.
        i += 1


def test_trigonometric_rational():
    g = env_trigonometric_rational()
    time_axis = []
    x1 = []
    x2 = []
    for t in range(1000):
        v1, v2 = next(g)
        time_axis.append(t)
        x1.append(v1)
        x2.append(v2)
    pyplot.plot(time_axis, x1)
    pyplot.plot(time_axis, x2)
    pyplot.show()


def test_ascending_descending_nominal():
    g = env_ascending_descending_nominal()
    for t in range(1000):
        print(next(g), end="")


if __name__ == "__main__":
    test_env_crypto()
    # test_text()
