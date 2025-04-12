#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
student_agent.py

此 Agent 會讀取 value.pkl 並利用 NTupleApproximator 做決策。
"""

import random
import pickle
import numpy as np
import common  # 載入共用模組

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

# 建立 approximator 與環境物件（用於模擬動作）
try:
    with open("value.pkl", "rb") as f:
        loaded_weights = pickle.load(f)
    approximator = common.NTupleApproximator(board_size=4, patterns=pattern, init_value=320000, use_tc=True)
    approximator.weights = loaded_weights
    print("Weights loaded successfully from value.pkl.")
except Exception as e:
    print("Warning: Unable to load value.pkl, using default initial weights. Error:", e)
    approximator = common.NTupleApproximator(board_size=4, patterns=pattern, init_value=320000, use_tc=True)

env_sim = common.Game2048Env(size=4)

def simulate_move(board, action):
    return env_sim.simulate_move(board, action)

def is_move_legal(board, action):
    return env_sim.is_move_legal(board, action)

def get_action(state, score):
    """
    接收狀態 state 與 score，返回一個動作（0~3）。
    依據所有合法動作模擬後，選擇使 NTupleApproximator 評估值最高的動作。
    """
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

# 下方測試區塊可在本地測試時執行，評分時系統只會呼叫 get_action 函數
if __name__ == "__main__":
    # 建立一個簡單初始棋盤作測試
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
