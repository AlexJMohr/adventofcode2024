#!/usr/bin/env python3

from dataclasses import dataclass
import itertools
import operator
import re
from collections import defaultdict, deque
from functools import cmp_to_key
from itertools import product, combinations

import click
import numpy as np
import sympy


@click.group()
def cli():
    pass


@cli.command()
@click.argument("file", type=click.File())
def day01(file):
    def parse(file):
        xs = []
        ys = []
        for line in file:
            x, y = re.search(r"(\d+)\s+(\d+)", line).groups()
            xs.append(int(x))
            ys.append(int(y))
        return xs, ys

    def distance(x, y):
        return abs(x - y)

    def part1(xs, ys):
        xs.sort()
        ys.sort()
        total = sum(distance(x, y) for x, y in zip(xs, ys))
        print("Part 1:", total)

    def similarity(x, ys):
        return x * len([y for y in ys if y == x])

    def part2(xs, ys):
        total = sum(similarity(x, ys) for x in xs)
        print("Part 2:", total)

    xs, ys = parse(file)
    part1(xs, ys)
    part2(xs, ys)


@cli.command()
@click.argument("file", type=click.File())
def day03(file):
    def part1(contents):
        pattern = re.compile(r"mul\((\d+),(\d+)\)")
        muls = pattern.findall(contents)
        total = sum(int(x) * int(y) for x, y in muls)
        print("Part 1:", total)

    def part2(contents):
        pattern = re.compile(r"(do|don't|mul)\((?:(\d+),(\d+))?\)")
        matches = pattern.findall(contents)

        total = 0
        do = True
        for instruction, x, y in matches:
            if instruction == "do":
                do = True
            elif instruction == "don't":
                do = False
            elif instruction == "mul":
                if do:
                    x = int(x)
                    y = int(y)
                    total += x * y

        print("Part 2:", total)

    contents = file.read()
    part1(contents)
    part2(contents)


def grid2str(grid):
    lines = []
    for row in grid:
        lines.append("".join(row))
    return "\n".join(lines)


@cli.command()
@click.argument("file", type=click.File())
def day04(file):
    def count_horizontal(grid):
        return len(re.findall(r"XMAS", grid2str(grid)))

    def count_diagonal(grid):
        rows, cols = grid.shape
        count = 0
        for row in range(rows - 3):
            for col in range(cols - 3):
                if (
                    grid[row, col] == "X"
                    and grid[row + 1, col + 1] == "M"
                    and grid[row + 2, col + 2] == "A"
                    and grid[row + 3, col + 3] == "S"
                ):
                    count += 1
        return count

    def part1(contents):
        # horizontal
        total = count_horizontal(np.copy(grid))
        total += count_horizontal(np.fliplr(np.copy(grid)))
        # vertical by transposing the grid
        total += count_horizontal(np.transpose(np.copy(grid)))
        total += count_horizontal(np.fliplr(np.transpose(np.copy(grid))))
        # diagonal
        total += count_diagonal(np.copy(grid))
        total += count_diagonal(np.fliplr(np.copy(grid)))
        total += count_diagonal(np.flipud(np.copy(grid)))
        total += count_diagonal(np.flipud(np.fliplr(np.copy(grid))))
        print("Part 1:", total)

    def part2(contents):
        pattern = np.array(
            [
                ["M", "", "M"],
                ["", "A", ""],
                ["S", "", "S"],
            ]
        )

        count = 0
        rows, cols = grid.shape
        for row in range(1, rows - 1):
            for col in range(1, cols - 1):
                if grid[row, col] == "A":
                    mini_grid = np.copy(grid[row - 1 : row + 2, col - 1 : col + 2])
                    # clear sides
                    mini_grid[0, 1] = ""
                    mini_grid[2, 1] = ""
                    mini_grid[1, 0] = ""
                    mini_grid[1, 2] = ""

                    if np.array_equal(mini_grid, pattern):
                        count += 1
                    elif np.array_equal(np.rot90(mini_grid, 1), pattern):
                        count += 1
                    elif np.array_equal(np.rot90(mini_grid, 2), pattern):
                        count += 1
                    elif np.array_equal(np.rot90(mini_grid, 3), pattern):
                        count += 1

        print("Part 2:", count)

    contents = file.read()
    grid = np.array([list(line) for line in contents.splitlines()])
    part1(np.copy(grid))
    part2(np.copy(grid))


@cli.command()
@click.argument("file", type=click.File())
def day05(file):
    def parse(contents):
        rules = set()
        updates = []
        lines = contents.splitlines()
        line_iter = iter(lines)
        for line in line_iter:
            if len(line) == 0:
                break
            x, y = line.split("|")
            rules.add((int(x), int(y)))
        for line in line_iter:
            updates.append([int(x) for x in line.split(",")])
        return rules, updates

    def is_correct(rules, update):
        for x, y in rules:
            try:
                if update.index(x) > update.index(y):
                    return False
            except ValueError:
                # if either value is not in the list, rule doesn't apply
                continue
        return True

    def total(updates):
        return sum(update[len(update) // 2] for update in updates)

    def part1(rules, updates):
        correct_updates = [update for update in updates if is_correct(rules, update)]
        print("Part 1:", total(correct_updates))

    def part2(rules, updates):
        def compare(a, b):
            if (a, b) in rules:
                return 1
            elif (b, a) in rules:
                return -1
            else:
                return 0

        corrected_updates = [
            sorted(update, key=cmp_to_key(compare))
            for update in updates
            if not is_correct(rules, update)
        ]
        print("Part 2:", total(corrected_updates))

    contents = file.read()
    rules, updates = parse(contents)
    part1(rules, updates)
    part2(rules, updates)


@cli.command()
@click.argument("file", type=click.File())
def day06(file):
    def turn_right(dir):
        return (-dir[1], dir[0])

    def parse(contents):
        grid = {}
        start_x, start_y = None, None
        for y, line in enumerate(contents.splitlines()):
            for x, char in enumerate(line):
                grid[(x, y)] = char
                if char == "^":
                    start_x, start_y = x, y
        return grid, start_x, start_y

    def walk(grid, start_x, start_y):
        # yield positions and directions
        x, y = start_x, start_y
        dir = (0, -1)
        while True:
            yield x, y, dir
            next_x, next_y = x + dir[0], y + dir[1]
            if (next_x, next_y) not in grid:
                break
            if grid[(next_x, next_y)] == "#":
                dir = turn_right(dir)
                next_x, next_y = x + dir[0], y + dir[1]
            else:
                x, y = next_x, next_y

    def part1(contents):
        grid, start_x, start_y = parse(contents)
        visited = {(x, y) for x, y, _ in walk(grid, start_x, start_y)}
        print("Part 1:", len(visited))

    def part2(contents):
        grid, start_x, start_y = parse(contents)

        loops = 0
        for coords, char in grid.items():
            if char != ".":
                continue
            g = grid.copy()
            g[coords] = "#"

            seen = set()
            for pos in walk(g, start_x, start_y):
                if pos in seen:
                    loops += 1
                    break
                seen.add(pos)

        # 551 too low
        # 1864 too low
        print("Part 2:", loops)

    contents = file.read()
    part1(contents)
    part2(contents)


@cli.command()
@click.argument("file", type=click.File())
def day07(file):
    def parse(contents):
        equations = []
        for line in contents.splitlines():
            test_val, rhs = line.split(":")
            test_val = int(test_val)
            inputs = [int(x) for x in re.findall(r"(\d+)", rhs)]
            equations.append((test_val, inputs))
        return equations

    def test_equations(equations, operators):
        total = 0
        for test_val, inputs in equations:
            for ops in product(operators, repeat=len(inputs) - 1):
                val, remaining_inputs = inputs[0], inputs[1:]
                for op, x in zip(ops, remaining_inputs):
                    val = op(val, x)
                if val == test_val:
                    total += test_val
                    break
        return total

    def part1(contents):
        equations = parse(contents)
        operators = [operator.add, operator.mul]
        total = test_equations(equations, operators)
        print("Part 1:", total)  # 2654749936343

    def part2(contents):
        def concat(x, y):
            power = 10
            while y >= power:
                power *= 10
            return x * power + y

        equations = parse(contents)
        operators = [operator.add, operator.mul, concat]
        total = test_equations(equations, operators)
        print("Part 2:", total)  # 124060392153684

    contents = file.read()
    part1(contents)
    part2(contents)


@dataclass(eq=True)
class Vec:
    x: int
    y: int

    def __add__(self, other):
        return Vec(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec(self.x - other.x, self.y - other.y)

    def __hash__(self):
        return hash((self.x, self.y))


@cli.command()
@click.argument("file", type=click.File())
def day08(file):
    def parse(contents):
        width, height = 0, 0
        char_positions = defaultdict(list)
        for y, line in enumerate(contents.splitlines()):
            height += 1
            width = len(line)
            for x, char in enumerate(line):
                if char != ".":
                    char_positions[char].append(Vec(x, y))

        return dict(char_positions), width, height

    def part1(contents):
        char_positions, width, height = parse(contents)
        antinode_locations = set()
        for char, positions in char_positions.items():
            for p, q in combinations(positions, r=2):
                delta = p - q
                pos = p + delta
                if pos.x >= 0 and pos.x < width and pos.y >= 0 and pos.y < height:
                    antinode_locations.add(pos)
                pos = q - delta
                if pos.x >= 0 and pos.x < width and pos.y >= 0 and pos.y < height:
                    antinode_locations.add(pos)

        print("Part 1:", len(antinode_locations))  # 371

    def part2(contents):
        char_positions, width, height = parse(contents)
        antinode_locations = set()
        for char, positions in char_positions.items():
            for p, q in combinations(positions, r=2):
                delta = q - p
                pos = p + delta
                while pos.x >= 0 and pos.x < width and pos.y >= 0 and pos.y < height:
                    antinode_locations.add(pos)
                    pos += delta
                pos = q - delta
                while pos.x >= 0 and pos.x < width and pos.y >= 0 and pos.y < height:
                    antinode_locations.add(pos)
                    pos -= delta

        print("Part 2:", len(antinode_locations))  # 1229

    contents = file.read()
    part1(contents)
    part2(contents)


@cli.command()
@click.argument("file", type=click.File())
def day09(file):
    def calculate_checksum(disk):
        checksum = 0
        for i, c in enumerate(disk):
            if c != ".":
                checksum += int(c) * i
        return checksum

    def part1(contents):
        disk = []
        id_sequence = itertools.count()
        for i, c in enumerate(contents):
            length = int(c)
            if i % 2 != 0:
                disk.extend(["."] * length)
            else:
                disk.extend([str(next(id_sequence))] * length)

        right_index = len(disk) - 1
        for left_index in range(len(disk)):
            if disk[left_index] == ".":
                while disk[right_index] == ".":
                    right_index -= 1
                if right_index <= left_index:
                    break
                disk[left_index], disk[right_index] = (
                    disk[right_index],
                    disk[left_index],
                )

        print("Part 1:", calculate_checksum(disk))  # 6330095022244

    def part2(contents):

        @dataclass
        class DiskEntry:
            start: int
            length: int
            file_id: int | None = None

            def is_file(self):
                return self.file_id is not None

            def is_free(self):
                return self.file_id is None

            def __str__(self):
                return ("." if self.is_free() else str(self.file_id)) * self.length

            def __repr__(self):
                if self.is_free():
                    return f"Free(start={self.start}, length={self.length})"
                else:
                    return f"File(start={self.start}, length={self.length}, id={self.file_id})"

        def disk2str(disk):
            # disk.sort(key=lambda e: e.start)
            return "".join(str(e) for e in disk)

        def assert_disk_is_valid(disk):
            assert disk == sorted(disk, key=lambda e: e.start)
            start = 0
            for entry in disk:
                assert entry.start == start
                start += entry.length

        def print_disk(disk):
            print(disk2str(disk))

        disk: list[DiskEntry] = []
        start = 0
        file_id = -1
        for i, c in enumerate(contents):
            length = int(c)
            if i % 2 != 0:
                disk.append(DiskEntry(start=start, length=length))
            else:
                file_id += 1
                disk.append(DiskEntry(start=start, length=length, file_id=file_id))
            start += length

        assert_disk_is_valid(disk)
        # print_disk(disk)

        while file_id > 0:
            file = None
            file_idx = None
            for idx in range(len(disk) - 1, -1, -1):
                entry = disk[idx]
                if entry.file_id == file_id:
                    file = entry
                    file_idx = idx
                    break

            for free_idx, free in enumerate(disk):
                if (
                    free.is_free()
                    and free.length >= file.length
                    and free.start < file.start
                ):
                    # files up until where free space is
                    new_disk = disk[:free_idx]

                    # copy file to free space
                    new_disk.append(
                        DiskEntry(
                            start=free.start,
                            length=file.length,
                            file_id=file.file_id,
                        )
                    )

                    # remaining free space after file
                    if free.length > file.length:
                        new_disk.append(
                            DiskEntry(
                                start=free.start + file.length,
                                length=free.length - file.length,
                            )
                        )

                    # entries up until where file was
                    new_disk.extend(disk[free_idx + 1 : file_idx])

                    # replace original file with free space
                    new_disk.append(DiskEntry(start=file.start, length=file.length))

                    # remainder of disk
                    new_disk.extend(disk[file_idx + 1 :])

                    # for entry in new_disk:
                    #     print(" " * entry.start + str(entry))
                    # exit(1)

                    disk = new_disk
                    # print_disk(disk)
                    break

            file_id -= 1

        assert_disk_is_valid(disk)

        disk_list = []
        for entry in disk:
            disk_list.extend(["." if entry.is_free() else entry.file_id] * entry.length)
        print("Part 2:", calculate_checksum(disk_list))

    contents = file.read().strip()
    part1(contents)
    part2(contents)


@cli.command()
@click.argument("file", type=click.File())
def day10(file):
    def parse(contents):
        grid = {}
        trail_heads = []
        for y, line in enumerate(contents.splitlines()):
            for x, char in enumerate(line):
                height = int(char)
                grid[(x, y)] = height
                if height == 0:
                    trail_heads.append((x, y))
        return grid, trail_heads

    def part1(contents):
        grid, trail_heads = parse(contents)
        def count_trail_peaks(start):
            queue = deque([start])
            found_peaks = set()

            while queue:
                x, y = queue.popleft()
                if grid[(x, y)] == 9:
                    found_peaks.add((x, y))
                    continue
                neighbours = [
                    (x + 1, y),
                    (x - 1, y),
                    (x, y + 1),
                    (x, y - 1),
                ]
                for n in neighbours:
                    if n in grid and grid[(x, y)] + 1 == grid[n]:
                        queue.append(n)
            return len(found_peaks)

        total = 0
        for trail_head in trail_heads:
            total += count_trail_peaks(trail_head)
    
        print("Part 1:", total) # 709

    def part2(contents):
        grid, trail_heads = parse(contents)

        def get_num_trails(start):
            num_trails = 0
            queue = deque([start])
            while queue:
                x, y = queue.popleft()
                if grid[(x, y)] == 9:
                    num_trails += 1
                    continue
                neighbours = [
                    (x + 1, y),
                    (x - 1, y),
                    (x, y + 1),
                    (x, y - 1),
                ]
                for n in neighbours:
                    if n in grid and grid[(x, y)] + 1 == grid[n]:
                        queue.append(n)
            return num_trails
    
        total = 0
        for trail_head in trail_heads:
            # get the number of trails that begin at this trail head
            total += get_num_trails(trail_head)
        print("Part 2:", total)


    contents = file.read().strip()
    part1(contents)
    part2(contents)


@cli.command()
@click.argument("file", type=click.File())
def day11(file):
    def parse(contents):
        return [int(x) for x in contents.split()]
    
    def part1(contents):
        stones = parse(contents)
        for _ in range(25):
            new_stones = []
            for stone in stones:
                if stone == 0:
                    new_stones.append(1)
                elif len(str(stone)) % 2 == 0:
                    str_stone = str(stone)
                    half = len(str_stone) // 2
                    new_stones.append(int(str_stone[:half]))
                    new_stones.append(int(str_stone[half:]))
                else:
                    new_stones.append(stone * 2024)
            stones = new_stones

        print("Part 1:", len(stones)) # 189547
    
    def part2(contents):
        stones = parse(contents)
        stones = {stone: 1 for stone in stones}

        for _ in range(75):
            new_stones = defaultdict(int)
            for stone, count in stones.items():
                if stone == 0:
                    new_stones[1] += count
                elif len(str(stone)) % 2 == 0:
                    str_stone = str(stone)
                    half = len(str_stone) // 2
                    new_stones[int(str_stone[:half])] += count
                    new_stones[int(str_stone[half:])] += count
                else:
                    new_stones[stone * 2024] += count
            stones = new_stones
        
        print("Part 2:", sum(stones.values()))  # 224577979481346
    
    contents = file.read().strip()
    part1(contents)
    part2(contents)


@cli.command()
@click.argument("file", type=click.File())
def day12(file):
    contents = file.read().strip()

    grid = {}
    for row, line in enumerate(contents.splitlines()):
        for col, char in enumerate(line):
            grid[(row, col)] = char
    
    def flood(grid, start, target, visited):
        queue = deque([start])
        region = set()
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            region.add(current)
            x, y = current
            neighbours = [
                (x + 1, y),
                (x - 1, y),
                (x, y + 1),
                (x, y - 1),
            ]
            for n in neighbours:
                if n in grid and n not in visited and grid[n] == target:
                    queue.append(n)
        return region
    
    def find_perimeter(region):
        perimiter = 0
        for pos in region:
            x, y = pos
            neighbours = [
                (x + 1, y),
                (x - 1, y),
                (x, y + 1),
                (x, y - 1),
            ]
            for n in neighbours:
                if n not in region:
                    perimiter += 1
        return perimiter
    
    def count_edges(region):
        edges = 0
        for pos in region:
            x, y = pos
            north = (x, y - 1)
            east = (x + 1, y)
            south = (x, y + 1)
            west = (x - 1, y)
            north_west = (x - 1, y - 1)
            south_west = (x - 1, y + 1)
            north_east = (x + 1, y - 1)

            if north not in region and not (west in region and north_west not in region):
                edges += 1
            if south not in region and not (west in region and south_west not in region):
                edges += 1
            if west not in region and not (north in region and north_west not in region):
                edges += 1
            if east not in region and not (north in region and north_east not in region):
                edges += 1
            
        return edges

    visited = set()
    total_part1 = 0
    total_part2 = 0
    for pos in grid:
        if pos not in visited:
            target = grid[pos]
            region = flood(grid, pos, target, visited)
            area = len(region)
            perimiter = find_perimeter(region)
            edges = count_edges(region)
            total_part1 += area * perimiter
            total_part2 += area * edges

    print("Part 1:", total_part1) # 1533024
    print("Part 2:", total_part2) # 910066


@cli.command()
@click.argument("file", type=click.File())
def day13(file):
    contents = file.read().strip()

    machines = []

    a = None
    b = None
    for line in contents.splitlines():
        if line.startswith("Button"):
            x, y = [int(i) for i in re.findall(r"(\d+)", line)]
            x, y = int(x), int(y)
            if a is None:
                a = (x, y)
            else:
                b = (x, y)
        elif line.startswith("Prize"):
            prize_coords = tuple(int(i) for i in re.findall(r"(\d+)", line))
            machines.append((a, b, prize_coords))
            a = None
            b = None
    
    def find_steps(a, b, prize_coords):
        ax, ay = a
        bx, by = b
        px, py = prize_coords

        x, y = sympy.symbols("x y")
        solutions_x = sympy.diophantine(ax * x + bx * y - px)
        solutions_y = sympy.diophantine(ay * x + by * y - py)
        if not solutions_x or not solutions_y:
            return None
        solution_x = next(iter(solutions_x))
        solution_y = next(iter(solutions_y))
        return solution_x, solution_y
    
    for machine in machines:
        a, b, prize_coords = machine
        print(a, b, prize_coords)
        steps = find_steps(a, b, prize_coords)
        if steps is not None:
            print(steps)


if __name__ == "__main__":
    cli()
