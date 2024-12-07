#!/usr/bin/env python3

from functools import cmp_to_key
from itertools import product
import operator
import re

import click
import numpy as np


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


if __name__ == "__main__":
    cli()
