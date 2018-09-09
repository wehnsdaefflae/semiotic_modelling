#!/usr/bin/env python3
# coding=utf-8


def g():
    state = 0
    y_value = state % 2
    while True:
        s_value = yield y_value
        state += s_value
        y_value = state % 2


if __name__ == "__main__":
    g = g()
    pass
