#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
student_agent.py

說明：
1. 此程式碼讀取已訓練好的權重檔案 (value.pkl) 並建立 NTupleApproximator。
2. 實作了 get_action(state, score) 函數，依據目前 board (state) 模擬所有合法動作，
   並選擇使估值最大的動作 (0: up, 1: down, 2: left, 3: right)。
3. 請確認 "value.pkl" 檔案與本程式置於同一目錄，否則程式將以預設初始權重進行。
"""

import math
import random
import pickle
import numpy as np
from collections import defaultdict

##############################################
# ConstantFactory：用於 defaultdict 的預設值建立
##############################################
class ConstantFactory:
    def __init__(self, value):
        self.value = value
    def __call__(self):
        return self.value

##############################################
# 2048 遊戲環境 (只包含模擬需要的部份，不包含隨機 tile 加入)
##############################################
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
        """
        模擬走一步，不加隨機 tile。
        action: 0: up, 1: down, 2: left, 3: right
        """
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

##############################################
# NTupleApproximator 及相關對稱變換函式
##############################################
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

class NTupleApproximator:
    def __init__(self, board_size, patterns, init_value=320000, use_tc=False):
        self.board_size = board_size
        self.patterns = patterns
        self.use_tc = use_tc
        num_patterns = len(self.patterns)
        self.init_val_per_pattern = init_value / num_patterns if num_patterns > 0 else 0.0
        # 使用 defaultdict 與 ConstantFactory 來初始化權重
        self.weights = [defaultdict(ConstantFactory(self.init_val_per_pattern)) for _ in patterns]
        if self.use_tc:
            self.tc_E = [defaultdict(float) for _ in patterns]
            self.tc_A = [defaultdict(float) for _ in patterns]

        # 產生所有對稱變換
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
        # 0 對應 0，其它 tile 使用 log2 得到指數
        return 0 if tile == 0 else int(math.log(tile, 2))

    def get_feature(self, board, coords):
        flat = board.flatten() if isinstance(board, np.ndarray) and board.ndim > 1 else board
        return tuple(self.tile_to_index(flat[i]) for i in coords)

    def value(self, board):
        total = 0.0
        num = len(self.patterns)
        for i in range(num):
            # 對應第 i 模式，使用所有對稱版型
            group = self.symmetry_patterns[i * self.sym_per_pattern:(i + 1) * self.sym_per_pattern]
            val = 0.0
            for pat in group:
                feat = self.get_feature(board, pat)
                val += self.weights[i][feat]
            total += val / len(group)
        return total

    def update(self, board, delta, alpha):
        # 此範例中未於 get_action 中使用 update
        num = len(self.patterns)
        for i in range(num):
            group = self.symmetry_patterns[i * self.sym_per_pattern:(i + 1) * self.sym_per_pattern]
            for pat in group:
                feat = self.get_feature(board, pat)
                self.weights[i][feat] += alpha * (delta / len(group))

##############################################
# 定義 2048 常用的 n-tuple 模式
##############################################
pattern = [
    [0, 1, 2, 4, 5, 6],
    [1, 2, 5, 6, 9, 13],
    [0, 1, 2, 3, 4, 5],
    [0, 1, 5, 6, 7, 10],
    [0, 1, 2, 5, 9, 10],
    [0, 1, 5, 9, 13, 14],
    [0, 1, 5, 8, 9, 13],
    [0, 1, 2, 4, 6, 10],
]

##############################################
# 載入訓練後的權重 (value.pkl)
##############################################
# 建立 approximator 的全域變數，供 get_action 使用
try:
    with open("value.pkl", "rb") as f:
        loaded_weights = pickle.load(f)
    approximator = NTupleApproximator(board_size=4, patterns=pattern, init_value=320000, use_tc=True)
    approximator.weights = loaded_weights
    print("Weights loaded successfully from value.pkl.")
except Exception as e:
    print("Warning: Unable to load value.pkl, using default initial weights. Error:", e)
    approximator = NTupleApproximator(board_size=4, patterns=pattern, init_value=320000, use_tc=True)

# 建立環境物件 (用於模擬走步與檢查合法動作)
env_sim = Game2048Env(size=4)

##############################################
# 輔助函式：模擬走一步與判斷動作是否合法
##############################################
def simulate_move(board, action):
    return env_sim.simulate_move(board, action)

def is_move_legal(board, action):
    return env_sim.is_move_legal(board, action)

##############################################
# 核心函式：get_action(state, score)
##############################################
def get_action(state, score):
    """
    傳入 state (4x4 ndarray) 與目前分數 (score)，返回一個動作 (0~3)
    策略：對所有合法動作，模擬走一步並使用 NTupleApproximator 計算估值，
    選擇估值最高的動作。如果都不合法 (非常罕見)，則隨機選一個動作。
    """
    legal_actions = [a for a in range(4) if is_move_legal(state, a)]
    if not legal_actions:
        # 若無合法動作，回傳預設動作（例如 0）
        return 0

    best_action = legal_actions[0]
    best_value = -float('inf')
    
    for action in legal_actions:
        sim_state = simulate_move(state, action)
        value_est = approximator.value(sim_state)
        # 若評估值更好則更新最佳動作
        if value_est > best_value:
            best_value = value_est
            best_action = action

    return best_action

##############################################
# 測試區塊 (僅供本地測試用，不影響評分系統)
##############################################
if __name__ == "__main__":
    # 建立一個隨機初始棋盤
    state = np.zeros((4, 4), dtype=int)
    # 隨機加入兩個 tile (2 或 4)
    empty_cells = list(zip(*np.where(state == 0)))
    for _ in range(2):
        if empty_cells:
            x, y = random.choice(empty_cells)
            state[x, y] = 2 if random.random() < 0.9 else 4
            empty_cells = list(zip(*np.where(state == 0)))
    score = 0
    print("Initial board:")
    print(state)
    action = get_action(state, score)
    action_str = ["up", "down", "left", "right"][action]
    print("Chosen action:", action, action_str)
