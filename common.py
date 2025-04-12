# common.py
import math
import random
import numpy as np
from collections import defaultdict

############################
# ConstantFactory 類別
############################
class ConstantFactory:
    def __init__(self, value):
        self.value = value
    def __call__(self):
        return self.value

############################
# 2048 遊戲環境相關函式
############################
class Game2048Env:
    def __init__(self, size=4):
        self.size = size

    def compress(self, row):
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def merge(self, row):
        for i in range(len(row) - 1):
            if row[i] == row[i+1] and row[i] != 0:
                row[i] *= 2
                row[i+1] = 0
        return row

    def simulate_move(self, board, action):
        """模擬動作，不加入新的 tile"""
        board_copy = board.copy()
        size = self.size
        if action == 0:  # up
            for j in range(size):
                col = board_copy[:, j]
                col = self.compress(col)
                col = self.merge(col)
                col = self.compress(col)
                board_copy[:, j] = col
        elif action == 1:  # down
            for j in range(size):
                col = board_copy[:, j][::-1].copy()
                col = self.compress(col)
                col = self.merge(col)
                col = self.compress(col)
                board_copy[:, j] = col[::-1]
        elif action == 2:  # left
            for i in range(size):
                row = board_copy[i]
                row = self.compress(row)
                row = self.merge(row)
                row = self.compress(row)
                board_copy[i] = row
        elif action == 3:  # right
            for i in range(size):
                row = board_copy[i][::-1].copy()
                row = self.compress(row)
                row = self.merge(row)
                row = self.compress(row)
                board_copy[i] = row[::-1]
        else:
            raise ValueError("Invalid action")
        return board_copy

    def is_move_legal(self, board, action):
        simulated = self.simulate_move(board, action)
        return not np.array_equal(simulated, board)

############################
# 對稱變換函式
############################
def identity(coord, board_size):
    return coord

def rot90(coord, board_size):
    r, c = coord
    return (c, board_size - 1 - r)

def rot180(coord, board_size):
    r, c = coord
    return (board_size - 1 - r, board_size - 1 - c)

def rot270(coord, board_size):
    r, c = coord
    return (board_size - 1 - c, r)

def reflect_horizontal(coord, board_size):
    r, c = coord
    return (r, board_size - 1 - c)

def reflect_vertical(coord, board_size):
    r, c = coord
    return (board_size - 1 - r, c)

def reflect_main(coord, board_size):
    r, c = coord
    return (c, r)

def reflect_anti(coord, board_size):
    r, c = coord
    return (board_size - 1 - c, board_size - 1 - r)

############################
# NTupleApproximator 類別
############################
class NTupleApproximator:
    def __init__(self, board_size, patterns, init_value=320000, use_tc=False):
        self.board_size = board_size
        self.patterns = patterns
        self.use_tc = use_tc
        num_patterns = len(self.patterns)
        self.init_val_per_pattern = init_value / num_patterns if num_patterns > 0 else 0.0
        self.weights = [defaultdict(ConstantFactory(self.init_val_per_pattern)) for _ in patterns]
        if self.use_tc:
            self.tc_E = [defaultdict(float) for _ in patterns]
            self.tc_A = [defaultdict(float) for _ in patterns]

        # 產生所有對稱變換版本
        self.symmetry_patterns = []
        for p in self.patterns:
            syms = self.generate_symmetries(p)
            for s in syms:
                self.symmetry_patterns.append(s)
        self.sym_per_pattern = 8 if num_patterns > 0 else 0

    def generate_symmetries(self, pattern):
        board_size = self.board_size
        funcs = [identity, rot90, rot180, rot270, reflect_horizontal, reflect_vertical, reflect_main, reflect_anti]
        sym_patterns = []
        for f in funcs:
            new_p = []
            for idx in pattern:
                r = idx // board_size
                c = idx % board_size
                nr, nc = f((r, c), board_size)
                new_p.append(nr * board_size + nc)
            if new_p not in sym_patterns:
                sym_patterns.append(new_p)
        return sym_patterns

    def tile_to_index(self, tile):
        return 0 if tile == 0 else int(math.log(tile, 2))

    def get_feature(self, board, coords):
        flat = board.flatten() if isinstance(board, np.ndarray) and board.ndim > 1 else board
        return tuple(self.tile_to_index(flat[i]) for i in coords)

    def value(self, board):
        total = 0.0
        num = len(self.patterns)
        for i in range(num):
            group = self.symmetry_patterns[i * self.sym_per_pattern:(i + 1) * self.sym_per_pattern]
            val = 0.0
            for pat in group:
                feat = self.get_feature(board, pat)
                val += self.weights[i][feat]
            total += val / len(group)
        return total

    def update(self, board, delta, alpha):
        num = len(self.patterns)
        for i in range(num):
            group = self.symmetry_patterns[i * self.sym_per_pattern:(i + 1) * self.sym_per_pattern]
            for pat in group:
                feat = self.get_feature(board, pat)
                self.weights[i][feat] += alpha * (delta / len(group))
