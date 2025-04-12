import math
import copy
import random
import pickle
import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
from collections import defaultdict

# 頂層定義 ConstantFactory 類別，用來作為 defaultdict 的預設值工廠
class ConstantFactory:
    def __init__(self, value):
        self.value = value
    def __call__(self):
        return self.value

##############################################
# 以下這段是 Game2048Env 的程式碼 (不做修改)
##############################################
class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()
        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True  # Record if the last move was valid

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left"""
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row"""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        """Move the board left"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        """Move the board right"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        """Move the board up"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        """Move the board down"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        """Check if there are no legal moves left"""
        if np.any(self.board == 0):
            return False
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False
        return True

    def step(self, action):
        """Execute one action"""
        assert self.action_space.contains(action), "Invalid action"
        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved
        if moved:
            self.add_random_tile()
        done = self.is_game_over()
        return self.board, self.score, done, {}

    def render(self, mode="human", action=None):
        """Render the current board using Matplotlib."""
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)
        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)
                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        """Simulate a left move for a single row"""
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i+1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i+1] = 0
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        """Check if the specified move is legal (i.e., changes the board)"""
        temp_board = self.board.copy()
        if action == 0:
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:
            for j in range(self.size):
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")
        return not np.array_equal(self.board, temp_board)


##############################################
# 以下為 NTupleApproximator 與一些輔助函式（使用預設 pkl 權重）
##############################################
# 對稱變換函式
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

# 預設的 n-tuple 模式
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

class NTupleApproximator:
    def __init__(self, board_size, patterns, init_value=320000, use_tc=False):
        self.board_size = board_size
        self.patterns = patterns
        self.use_tc = use_tc
        num_patterns = len(self.patterns)
        self.init_val_per_pattern = init_value / num_patterns if num_patterns > 0 else 0.0
        # 使用 defaultdict 搭配頂層 ConstantFactory 確保 pickle 時能正常序列化
        self.weights = [defaultdict(ConstantFactory(self.init_val_per_pattern)) for _ in patterns]
        if self.use_tc:
            self.tc_E = [defaultdict(float) for _ in patterns]
            self.tc_A = [defaultdict(float) for _ in patterns]
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
            n_sym = len(group)
            for pat in group:
                feat = self.get_feature(board, pat)
                if self.use_tc:
                    curE = self.tc_E[i].get(feat, 0.0)
                    curA = self.tc_A[i].get(feat, 0.0)
                    self.tc_E[i][feat] = curE + delta
                    self.tc_A[i][feat] = curA + abs(delta)
                    beta = abs(self.tc_E[i][feat]) / self.tc_A[i][feat] if self.tc_A[i][feat] != 0 else 1.0
                    curr = self.weights[i][feat]
                    self.weights[i][feat] = curr + alpha * beta * (delta / n_sym)
                else:
                    curr = self.weights[i][feat]
                    self.weights[i][feat] = curr + alpha * (delta / n_sym)

##############################################
# 以下為輔助函式與 TD-MCTS 的部分 (略)
##############################################
# 此處略過 TD-MCTS 與 rollout 等函式，請根據需要整合你的程式

##############################################
# 載入預訓練權重並測試 get_action
##############################################
# 以下僅為示範，請確認你的 "value.pkl" 檔案在正確的位置
if __name__ == "__main__":
    # 測試載入預訓練權重 (value.pkl) 並執行環境測試
    approxi = NTupleApproximator(board_size=4, patterns=pattern, init_value=320000, use_tc=True)
    try:
        with open("value.pkl", "rb") as f:
            approxi.weights = pickle.load(f)
        print("Weights loaded successfully.")
    except Exception as e:
        print("Error loading weights:", e)

    # 建立環境並進行簡單測試
    env = Game2048Env()
    state = env.reset()
    env.render()
    print("NTupleApproximator value of initial state:", approxi.value(state))
