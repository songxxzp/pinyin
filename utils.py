import os

from typing import Tuple, List


def extract_path(path: str) -> List:
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


def extract_data_path(data_tuple_list: List[Tuple]) -> List[Tuple]:
    extracted_data_tuple_list = []
    for path, format, labels in data_tuple_list:
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), path)
        if os.path.isdir(path):
            for file_path in extract_path(path):
                extracted_data_tuple_list.append((file_path, format, labels))
        else:
            extracted_data_tuple_list.append((path, format, labels))
    return extracted_data_tuple_list
