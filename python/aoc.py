#!/usr/bin/env python3

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
        for (instruction, x, y) in matches:
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


@cli.command()
@click.argument("file", type=click.File())
def day04(file):
    def grid2str(grid):
        lines = []
        for row in grid:
            lines.append("".join(row))
        return "\n".join(lines)

    def count_horizontal(grid):
        return len(re.findall(r"XMAS", grid2str(grid)))
    
    def count_diagonal(grid):
        rows, cols = grid.shape
        count = 0
        for row in range(rows - 3):
            for col in range(cols - 3):
                if grid[row, col] == "X" and grid[row+1, col+1] == "M" and grid[row+2, col+2] == "A" and grid[row+3, col+3] == "S":
                    count += 1
        return count
    
    contents = file.read()
    grid = np.array([list(line) for line in contents.splitlines()])

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
    print(total)



if __name__ == "__main__":
    cli()
