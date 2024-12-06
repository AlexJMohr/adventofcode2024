#!/usr/bin/env python3

from collections import defaultdict
from functools import cmp_to_key
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
        grid = np.array([list(line) for line in contents.splitlines()])
        starting_coords = np.where(grid == "^")
        x, y = starting_coords[1][0], starting_coords[0][0]
        return grid, x, y

    def part1(grid, x, y):
        visited = set()
        visited.add((x, y))

        dir = (0, -1)  # up
        width, height = grid.shape
        while True:
            next_x = x + dir[0]
            next_y = y + dir[1]
            if next_y < 0 or next_y >= height or next_x < 0 or next_x >= width:
                break
            elif grid[next_y, next_x] == "#":
                dir = turn_right(dir)
            else:
                x, y = next_x, next_y
                visited.add((x, y))
                grid[y, x] = "X"
        print("Part 1:", len(visited))

    def part2(grid, x, y):
        width, height = grid.shape
        dir = (0, -1)
        visited = defaultdict(set)

        def walk_backwards(x, y, dir):
            # walk backwards from current position until the edge or "#", and mark the current direction
            while x >= 0 and x < width and y >= 0 and y < height and grid[y, x] != "#":
                visited[(x, y)].add(dir)
                # grid[y, x] = "X"
                x -= dir[0]
                y -= dir[1]

        walk_backwards(x, y, dir)

        obstacle_positions = set()
        while True:
            next_x = x + dir[0]
            next_y = y + dir[1]

            # check if we're about to walk off the grid
            if next_y < 0 or next_y >= height or next_x < 0 or next_x >= width:
                break

            # every time we turn, walk backwards and mark the new direction
            if grid[next_y, next_x] == "#":
                dir = turn_right(dir)
                walk_backwards(x, y, dir)

            next_x = x + dir[0]
            next_y = y + dir[1]

            if visited_dirs := visited.get((x, y)):
                # We're on a previously visited spot.
                # If we turn right, have we already gone that direction?
                # If so, we could put an obstacle in front of us.
                if turn_right(dir) in visited_dirs:
                    obstacle_positions.add((next_x, next_y))
                    grid[next_y, next_x] = "O"

            visited[(x, y)].add(dir)
            x, y = next_x, next_y

        # 551 too low
        print("Part 2:", len(obstacle_positions))

    contents = file.read()
    part1(*parse(contents))
    part2(*parse(contents))


if __name__ == "__main__":
    cli()
