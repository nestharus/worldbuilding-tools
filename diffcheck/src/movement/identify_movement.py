import json

from linkeds import DynamicList, DoubleNode

from matcher.longest_matching_blocks import find_longest_matching_blocks
from matcher.match import Match


def get_left_blocks(blocks):
    linked_list = DynamicList()
    for (left_start, _), _ in blocks:
        linked_list.add_last([left_start])
    return linked_list


def get_right_blocks(blocks):
    right_blocks = [tuple((right_start, left_start)) for (left_start, _), (right_start, _) in blocks]
    right_blocks.sort(key=lambda x: x[0])
    return right_blocks


def get_right_block_map(right_blocks, size):
    mapping = {}

    for index in range(len(right_blocks)):
        mapping[right_blocks[index][1]] = index

    return mapping


def get_new_neighbor(index, blocks):
    if index == 0:
        return None

    return blocks[index - 1][1]


def get_current_neighbor(node):
    neighbor = node.prev

    if neighbor is None:
        return None

    return neighbor._data[-1]


def get_new_home(new_neighbor, left_block_map):
    if new_neighbor is None:
        return None

    return left_block_map[new_neighbor]


def remove_node(linked_list, node):
    if node.next is None:
        linked_list._tail = node.prev
    else:
        node.next.prev = node.prev
    if node.prev is None:
        linked_list._head = node.next
    else:
        node.prev.next = node.next


def set_first_node(linked_list, node):
    node.next = linked_list._head
    linked_list._head.prev = node
    linked_list._head = node


def get_block_map(blocks, size):
    block_map = {}

    for block in blocks:
        block_map[block[0][0]] = block[0]

    return block_map


def get_left_block_map(left_blocks, size):
    mapping = {}

    node = left_blocks._head
    while node is not None:
        mapping[node._data[0]] = node
        node = node.next

    return mapping


def get_left_start_from_node(node) -> any:
    return node._data[0]


def get_right_start_from_node(node: DoubleNode, right_blocks: list[any], right_block_map: dict[int, any]):
    return right_blocks[right_block_map[get_left_start_from_node(node)]][0]


def identify_moved_blocks(left: list, right: list) -> list[tuple[int, int]]:
    blocks: list = find_longest_matching_blocks(left, right)
    left_blocks: DynamicList = get_left_blocks(blocks)
    right_blocks: list = get_right_blocks(blocks)

    block_map: dict = get_block_map(blocks, len(left))
    right_block_map: dict = get_right_block_map(right_blocks, len(left))
    left_block_map: dict = get_left_block_map(left_blocks, len(left))

    moved_nodes = []

    current_node = left_blocks._head
    while current_node is not None:
        next_node = current_node.next

        left_start = get_left_start_from_node(current_node)
        right_start = get_right_start_from_node(current_node, right_blocks, right_block_map)
        new_neighbor = get_new_neighbor(right_block_map[left_start], right_blocks)
        current_neighbor = get_current_neighbor(current_node)

        if new_neighbor is not current_neighbor and left_start != right_start:
            new_node = get_new_home(new_neighbor, left_block_map)

            remove_node(left_blocks, current_node)
            moved_nodes.append(current_node._data[0])

            if new_node is None:
                set_first_node(left_blocks, current_node)
            else:
                new_node._data.extend(current_node._data)
                left_block_map[current_node._data[-1]] = new_node

        current_node = next_node

    moved_nodes = [block_map[left_start] for left_start in moved_nodes]

    return moved_nodes


def to_moved_matches(blocks: list[tuple[int, int]], matches: list[Match]) -> list[Match]:
    return [
        matches[index]
        for index in Match.blocks_to_indices(blocks)
    ]


def to_unmoved_matches(
    blocks: list[tuple[int, int]],
    left_blocks: list[any],
    matches: list[Match]
) -> list[Match]:
    movement_set = {
        left_blocks[index]
        for index in Match.blocks_to_indices(blocks)
    }
    return [
        matches[index]
        for index, block in enumerate(left_blocks)
        if block not in movement_set
    ]