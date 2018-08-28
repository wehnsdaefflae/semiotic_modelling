import string
from typing import Generator, Union

from dateutil import parser
from matplotlib import pyplot

from data.data_processing import equisample, series_generator


def text_generator(file_path: str) -> Generator[str, None, None]:
    permissible_non_letter = string.digits + string.punctuation + " "
    with open(file_path, mode="r") as file:
        for line in file:
            for character in line:
                if character in string.ascii_letters:
                    yield character.lower()

                elif character in permissible_non_letter:
                    yield character


def crypto_generator(file_path: str, start_val: Union[int, str], end_val: Union[int, str], interval_seconds: int) -> Generator[float, None, None]:
    type_set = {type(start_val), type(end_val)}
    assert len(type_set) == 1
    t, = type_set

    if t == int:
        start_ts = start_val
        end_ts = end_val

    elif t == str:
        start_time = parser.parse(start_val)
        end_time = parser.parse(end_val)

        start_ts = start_time.timestamp()
        end_ts = end_time.timestamp()

    else:
        raise ValueError()

    for t, value in equisample(series_generator(file_path, start_timestamp=start_ts, end_timestamp=end_ts), target_delta=interval_seconds):
        yield value


def test_text():
    text_path = "D:/Data/Texts/pride_prejudice.txt"
    for v in text_generator(text_path):
        print(v, end="")


def test_rate():
    rate_path = "D:/Data/binance/01Jan2010--1m/EOSETH.csv"
    delta = 60

    start_time = 1501113720
    end_time = 1532534580

    time_axis = []
    value_axis = []

    for v in crypto_generator(rate_path, start_time, end_time, delta):
        time_axis.append(start_time)
        start_time += delta
        value_axis.append(v)

    pyplot.plot(time_axis, value_axis)
    pyplot.show()


if __name__ == "__main__":
    test_rate()
    # test_text()
