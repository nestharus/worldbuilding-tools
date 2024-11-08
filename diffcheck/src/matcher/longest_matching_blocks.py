import json


def find_match(
    left_index: int,
    right_index: int,
    left_elements: list,
    right_elements: list
) -> tuple[int, int]:
    left_element = left_elements[left_index]
    right_index = right_elements.index(left_element)
    length = 0

    while left_elements[left_index] == right_elements[right_index]:
        length += 1

    return right_index, length


def find_longest_matching_blocks(
    left_elements: list,
    right_elements: list
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    matching_blocks = []

    left_index = 0
    while left_index < len(left_elements):
        element = left_elements[left_index]
        right_index = right_elements.index(element)

        length = 0
        while left_index + length < len(left_elements) and right_index + length < len(right_elements) and left_elements[left_index + length] == right_elements[right_index + length]:
            length += 1

        matching_blocks.append(tuple((
            tuple((left_index, left_index + length - 1)),
            tuple((right_index, right_index + length - 1))
        )))

        left_index += length

    return matching_blocks
