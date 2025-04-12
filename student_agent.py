#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
student_agent.py

此 Agent 讀取 value_full.pkl（儲存了完整的 NTupleApproximator 物件）後，
利用 NTupleApproximator 做決策。
方案2：在 pickle.load 前使用模組別名補丁，將 __main__.ConstantFactory 指定為 common.ConstantFactory。
系統會呼叫 get_action(state, score) 並根據當前 board 決定方向 (0: up, 1: down, 2: left, 3: right)。
"""

# 方案2：模組補丁
import __main__
from common import ConstantFactory, NTupleApproximator, Game2048Env
__main__.ConstantFactory = ConstantFactory
__main__.NTupleApproximator = NTupleApproximator
__main__.Game2048Env = Game2048Env
import pickle
import numpy as np
import random

# 定義 2048 常用的 n-tuple 模式（必須與訓練時一致）
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

# 嘗試讀取訓練過的完整 NTupleApproximator 物件（這裡檔名使用 value_full.pkl，請保持一致）
try:
    with open("./checkpoints/checkpoint_100.pkl", "rb") as f:
        approximator = pickle.load(f)
    print("完整的 approximator 物件已成功載入")
    weight_count = sum(len(w) for w in approximator.weights)
    print(f"總權重條目數: {weight_count}")
except Exception as e:
    print(f"無法載入 value_full.pkl，使用預設權重。錯誤：{e}")
    approximator = NTupleApproximator(board_size=4, patterns=pattern, init_value=320000, use_tc=True)

# 建立遊戲環境物件 (僅用於模擬走步及檢查合法動作)
env_sim = Game2048Env()

def simulate_move(board, action):
    sim_env = Game2048Env()
    sim_env.board = board.copy()
    if action == 0:
        sim_env.move_up()
    elif action == 1:
        sim_env.move_down()
    elif action == 2:
        sim_env.move_left()
    elif action == 3:
        sim_env.move_right()
    return sim_env.board

def is_move_legal(board, action):
    sim_env = Game2048Env()
    sim_env.board = board.copy()
    return sim_env.is_move_legal(action)

def get_action(state, score):
    legal_actions = [a for a in range(4) if is_move_legal(state, a)]
    if not legal_actions:
        return 0
    best_action = legal_actions[0]
    best_value = -float('inf')
    for action in legal_actions:
        sim_state = simulate_move(state, action)
        value_est = approximator.value(sim_state)
        if value_est > best_value:
            best_value = value_est
            best_action = action
    return best_action

# 測試區塊 (僅供本地測試使用，評分系統只會呼叫 get_action)
if __name__ == "__main__":
    state = np.zeros((4, 4), dtype=int)
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
