import datetime
from math import sin
from typing import Generator, Tuple, Iterator

from dateutil.tz import tzutc
from matplotlib import pyplot


def series_generator(file_path: str, start_timestamp: int = -1, end_timestamp: int = -1) -> Generator[Tuple[float, float], None, None]:
    print("Reading time series from {:s}...".format(file_path))

    with open(file_path, mode="r") as file:
        row_ts = -1
        for i, line in enumerate(file):
            row = line.strip().split("\t")
            row_ts = float(row[0]) / 1000.
            if -1 < start_timestamp:
                if start_timestamp < row_ts:
                    if i < 1:
                        first_date = datetime.datetime.fromtimestamp(row_ts, tz=tzutc())
                        start_time = datetime.datetime.fromtimestamp(start_timestamp, tz=tzutc())
                        msg = "Source {:s} starts after {:s} (ts {:f}) at {:s} (ts {:f})!"
                        raise ValueError(msg.format(file_path, str(start_time), start_timestamp, str(first_date), row_ts))
                elif row_ts < end_timestamp:
                    continue

            if -1 < end_timestamp < row_ts:
                break

            close = float(row[4])
            yield row_ts, close

        if row_ts < end_timestamp:
            last_date = datetime.datetime.fromtimestamp(row_ts, tz=tzutc())
            end_time = datetime.datetime.fromtimestamp(end_timestamp, tz=tzutc())
            msg = "Source {:s} ends before {:s} (ts {:f}) at {:s} (ts {:f})!"
            raise ValueError(msg.format(file_path, str(end_time), end_timestamp, str(last_date), row_ts))


def equisample(iterator: Iterator[Tuple[float, float]], target_delta: float) -> Generator[Tuple[float, float], None, None]:
    assert 0 < target_delta
    last_time = -1
    last_value = 0.
    for time_stamp, value in iterator:
        delta = time_stamp - last_time

        if delta < target_delta:
            continue

        elif delta == target_delta or last_time < 0:
            assert last_time < 0 or time_stamp == last_time + target_delta
            yield time_stamp, value

            last_value = value
            last_time = time_stamp

        else:
            value_change = (value - last_value) / delta
            no_intermediate_steps = round(delta // target_delta)
            for each_step in range(no_intermediate_steps):
                last_value += value_change
                last_time += target_delta
                yield last_time, last_value


def difference(source_generator: Generator[float, None, None]) -> Generator[float, None, None]:
    last_value = next(source_generator)
    for each_value in source_generator:
        yield each_value - last_value
        last_value = each_value


def my_normalization(source_generator: Generator[float, None, None], drag: int) -> Generator[float, None, None]:
    min_v = 0.
    max_v = 1.
    for each_value in source_generator:
        if max_v < each_value:
            max_v = each_value
            min_v = (drag * min_v + each_value) / (drag + 1)
        elif each_value < min_v:
            max_v = (drag * max_v + each_value) / (drag + 1)
            min_v = each_value
        else:
            max_v = (drag * max_v + each_value) / (drag + 1)
            min_v = (drag * min_v + each_value) / (drag + 1)
        yield (each_value - min_v) / (max_v - min_v)


def zscore_normalization(source_generator: Generator[float, None, None], drag: int) -> Generator[float, None, None]:
    mean = 0.
    deviation = 0.
    for each_value in source_generator:
        mean = (mean * drag + each_value) / (drag + 1)
        this_deviation = each_value - mean
        if deviation == 0.:
            yield float(each_value >= mean)
        else:
            yield this_deviation / deviation
        deviation = (deviation * drag + this_deviation) / (drag + 1)


if __name__ == "__main__":
    # time_stamp_test()
    X = range(1000)
    # Y = [sin(_x/10.) for _x in X]
    Y = list(my_normalization((sin(_x / 10.) for _x in range(1000)), 100))
    pyplot.plot(X, Y)
    pyplot.show()
    exit()
