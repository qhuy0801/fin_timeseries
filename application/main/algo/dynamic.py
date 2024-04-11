"""
Dynamic programming algorithms
"""
from typing import List, Callable, Generator


def dynamic_processing(arr: List, func: Callable) -> Generator:
    """
    Dynamic programming with single list (array)
    :param arr:
    :param func:
    :return:
    """
    memo = {}
    for i_i, e_i in enumerate(arr):
        for i_j, e_j in enumerate(arr):
            if i_i == i_j:
                continue
            if e_i not in memo:
                if e_j in memo:
                    if e_i in memo[e_j]:
                        continue
                    memo[e_j].append(e_i)
                    yield e_i, e_j, func(e_i, e_j)
                else:
                    memo[e_i] = [e_j]
                    yield e_i, e_j, func(e_i, e_j)
