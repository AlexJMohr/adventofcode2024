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
        pattern = np.array([
            ["M", "", "M"],
            ["", "A", ""],
            ["S", "", "S"],
        ])

        count = 0
        rows, cols = grid.shape
        for row in range(1, rows - 1):
            for col in range(1, cols - 1):
                if grid[row, col] == "A":
                    mini_grid = np.copy(grid[row-1:row+2, col-1:col+2])
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
        rules = []
        updates = []
        lines = contents.splitlines()
        line_iter = iter(lines)
        for line in line_iter:
            if len(line) == 0:
                break
            x, y = line.split("|")
            rules.append((int(x), int(y)))
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
        
    def part1(rules, updates):
        correct_updates = [update for update in updates if is_correct(rules, update)]
        total = 0
        for update in correct_updates:
            total += update[len(update)//2]
        print("Part 1:", total)
    
    def part2(rules, updates):
        # rules = sorted(rules)

        # combine the rules into a flat list to determine the order
        combined_rules = []
        for x, y in rules:
            combined_rules.append(x)
            combined_rules.append(y)
        combined_rules = sorted(combined_rules)

        corrected_updates = []
        for update in updates:
            if not is_correct(rules, update):
                for x, y in rules:
                    try:
                        x_idx = update.index(x)
                        y_idx = update.index(y)
                    except ValueError:
                        # rule doesn't apply
                        continue
                    if x_idx > y_idx:
                        update[x_idx], update[y_idx] = update[y_idx], update[x_idx]
                corrected_updates.append(update)
        
        for update in corrected_updates:
            print(update)
        total = 0
        for update in corrected_updates:
            total += update[len(update)//2]
        print("Part 2:", total)

    # 5021 too high
    contents = file.read()
    rules, updates = parse(contents)
    part1(rules, updates)
    part2(rules, updates)


if __name__ == "__main__":
    cli()
