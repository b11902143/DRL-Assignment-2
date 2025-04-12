#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
student_agent.py

此 Agent 讀取 value.pkl （透過 pickle）後，利用 NTupleApproximator 做決策。
方案2：在 pickle.load 之前，使用模組別名補丁，將 __main__.ConstantFactory 指定為 common.ConstantFactory。

系統會呼叫 get_action(state, score) 並根據當前 board 決定方向（0: up, 1: down, 2: left, 3: right）。
"""

# 方案2：模組補丁，修正 pickle.load 時找不到 ConstantFactory 的問題
import __main__
from common import ConstantFactory, NTupleApproximator, Game2048Env
__main__.ConstantFactory = ConstantFactory

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

# 嘗試讀取訓練過的權重檔案 value.pkl
try:
    with open("value.pkl", "rb") as f:
        loaded_weights = pickle.load(f)
    approximator = NTupleApproximator(board_size=4, patterns=pattern, init_value=320000, use_tc=True)
    approximator.weights = loaded_weights
    print("Weights loaded successfully from value.pkl.")
except Exception as e:
    print("Warning: Unable to load value.pkl, using default initial weights. Error:", e)
    approximator = NTupleApproximator(board_size=4, patterns=pattern, init_value=320000, use_tc=True)

# 建立遊戲環境物件 (僅用於模擬走步及檢查合法動作)
env_sim = Game2048Env()

def simulate_move(board, action):
    """
    模擬在 board 上執行動作 action 的結果，不會產生新的 tile。
    這裡採用 Game2048Env 中定義的 simulate_row_move 與 is_move_legal 方法，
    因此直接使用環境內部的邏輯（本例使用環境內 self.board 進行模擬）。
    """
    # 由於 student_agent 只用於決策，所以我們使用一份 board 複本來模擬
    sim_env = Game2048Env()
    sim_env.board = board.copy()
    # 利用原本的動作模擬（注意：這裡無隨機 tile 加入）
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
    """
    檢查在 board 上，動作 action 是否為合法動作（即執行後 board 有改變）。
    使用 Game2048Env 原生的 is_move_legal 方法（模擬動作）。
    """
    sim_env = Game2048Env()
    sim_env.board = board.copy()
    return sim_env.is_move_legal(action)

def get_action(state, score):
    """
    傳入目前遊戲狀態 state（4x4 ndarray）與分數 score，返回動作（0~3）。
    
    策略：檢查所有合法動作，模擬走一步，利用 NTupleApproximator 估算每個局面的價值，
    選擇使估值最高的那個動作。如果都不合法（極罕見），則隨機回傳一個動作。
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

# 測試區塊 (僅供本地測試使用，評分系統只會呼叫 get_action)
if __name__ == "__main__":
    # 建立一個簡單的初始棋盤
    state = np.zeros((4, 4), dtype=int)
    empty_cells = list(zip(*np.where(state == 0)))
    # 隨機加入兩個 tile (2 或 4)
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
