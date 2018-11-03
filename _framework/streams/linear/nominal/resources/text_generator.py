# coding=utf-8
import string
from typing import Generator


def sequence_nominal_text(file_path: str) -> Generator[str, None, None]:
    permissible_non_letter = string.digits + string.punctuation + " "
    while True:
        with open(file_path, mode="r") as file:
            for line in file:
                for character in line:
                    if character in string.ascii_letters:
                        yield character.lower()

                    elif character in permissible_non_letter:
                        yield character
