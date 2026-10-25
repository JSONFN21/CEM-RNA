# read the target structure and group the bases we choose together
from itertools import product
import numpy as np

unpaired_states = ('A', 'C', 'G', 'U')
paired_states = ('AU', 'UA', 'CG', 'GC', 'GU', 'UG')


# list all base combinations for a group this size
def mismatch_maker(size):
    return tuple(''.join(chars) for chars in product(unpaired_states, repeat=size))


mismatched_states = mismatch_maker(2)
trimismatched_states = mismatch_maker(3)


# keep the positions and allowed base choices for one group
class DesignVariable:
    def __init__(self, index, kind, positions, states):
        self.index = index
        self.kind = kind
        self.positions = positions
        self.states = states

    @property
    def n_states(self):
        return len(self.states)


# find each base's partner, using -1 when it has no partner
def parse_dot_bracket(structure):
    structure = structure.strip()
    if not structure:
        raise ValueError('target structure must not be empty')
    pairing_partners = np.full(len(structure), -1, dtype=int)
    stack = []
    for position, character in enumerate(structure):
        if character == '(':
            stack.append(position)
        elif character == ')':
            if not stack:
                raise ValueError(f"unmatched ')' at position {position + 1}")
            opening = stack.pop()
            pairing_partners[position] = opening
            pairing_partners[opening] = position
        elif character != '.':
            raise ValueError("only '.', '(' and ')' are supported")
    if stack:
        raise ValueError(f"unmatched '(' at position {stack[-1] + 1}")
    return pairing_partners


# group paired bases, loop neighbors, and unpaired bases in sequence order
# return the groups, each position's group, and the pairing partners
def build_coupled_design_variables(structure):
    structure = structure.strip()
    pairing_partners = parse_dot_bracket(structure)
    paired_groups = []
    coupled_groups = []
    assigned_positions = set()
    stack = []
    inner_loop = (-1, -1)

    # put available unpaired neighbors together so we can sample them as a group
    def add_coupled(positions):
        if (
            len(set(positions)) != len(positions)
            or any(position < 0 or position >= len(structure) for position in positions)
            or any(pairing_partners[position] >= 0 or position in assigned_positions for position in positions)
        ):
            return
        coupled_groups.append(positions)
        assigned_positions.update(positions)

    # closing a pair tells us which loop sits inside it
    for right, character in enumerate(structure):
        if character == '(':
            if stack:
                stack[-1][1] += 1
            stack.append([right, 0])
        elif character == ')':
            left, child_count = stack.pop()
            inner_left, inner_right = inner_loop
            if child_count == 0:
                add_coupled((left + 1, right - 1))
            elif child_count == 1:
                left_gap = inner_left - left - 1
                right_gap = right - inner_right - 1
                if left_gap == 1 and right_gap == 1:
                    add_coupled((left + 1, right - 1))
                elif left_gap == 1 and right_gap > 0:
                    add_coupled((left + 1, inner_right + 1, right - 1))
                elif right_gap == 1 and left_gap > 0:
                    add_coupled((left + 1, inner_left - 1, right - 1))
                elif left_gap > 1 and right_gap > 1:
                    add_coupled((left + 1, right - 1))
                    add_coupled((inner_left - 1, inner_right + 1))
            paired_groups.append((left, right))
            assigned_positions.update((left, right))
            inner_loop = (left, right)

    # put the groups in the same order as their first base
    blocks = []
    blocks.extend(('paired', positions, paired_states) for positions in paired_groups)
    for positions in coupled_groups:
        if len(positions) == 2:
            blocks.append(('mismatch', positions, mismatched_states))
        else:
            blocks.append(('trimismatch', positions, trimismatched_states))
    for position in range(len(structure)):
        if position not in assigned_positions:
            blocks.append(('unpaired', (position,), unpaired_states))
    blocks.sort(key=lambda block: (min(block[1]), block[1]))

    # build the groups and a quick way to find each base's group
    variables = []
    variable_of_position = np.full(len(structure), -1, dtype=int)
    for index, (kind, positions, states) in enumerate(blocks):
        variable = DesignVariable(index, kind, positions, states)
        variables.append(variable)
        for position in positions:
            variable_of_position[position] = index
    return variables, variable_of_position, pairing_partners
