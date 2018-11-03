# coding=utf-8
import datetime
from typing import Optional, Union, Generator, Tuple

from dateutil import parser
from dateutil.tz import tzutc

from data_generation.data_processing import equisample


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


def sequence_rational_crypto(file_path: str, interval_seconds: int,
                             start_val: Optional[Union[int, str]] = None,
                             end_val: Optional[Union[int, str]]= None) -> Generator[float, None, None]:
    start_ts = _convert_to_timestamp(start_val)
    end_ts = _convert_to_timestamp(end_val)

    while True:
        raw_generator = series_generator(file_path, start_timestamp=start_ts, end_timestamp=end_ts)
        for t, value in equisample(raw_generator, interval_seconds):
            yield value
