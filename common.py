# common.py
import math
import random
import numpy as np
from collections import defaultdict
import gym
from gym import spaces

# 頂層常數工廠，返回指定常數值
class ConstantFactory:
    def __init__(self, value):
        self.value = value
    def __call__(self):
        return self.value


class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()
        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True  # 記錄上一次移動是否合法

        self.reset()

    def reset(self):
        """重置環境"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        """在空格內隨機加入 tile (2 或 4)"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """壓縮一列：將非0數值向左靠"""
        new_row = row[row != 0]  # 移除0
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def merge(self, row):
        """合併相鄰相同的數字"""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        """向左移動棋盤"""
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
        """向右移動棋盤"""
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
        """向上移動棋盤"""
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
        """向下移動棋盤"""
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
        """檢查是否無合法移動"""
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
        """執行一次動作"""
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
        """使用 Matplotlib 呈現棋盤狀態"""
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
        """模擬對單一列執行向左移動"""
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
        """檢查指定移動是否合法"""
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


# ================= NTuple Approximator 定義 =================
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
        """
        初始化 N-Tuple approximator。
        :param board_size: 棋盤大小，例如 4
        :param patterns: 每個 n-tuple 模式（以棋盤位置 index 的 list 表示）
        :param init_value: 樂觀初始化時使用的總初始值，將會均分到各個模式
        :param use_tc: 是否使用 Temporal Coherence (TC) learning
        """
        self.board_size = board_size
        self.patterns = patterns
        self.use_tc = use_tc
        num_patterns = len(self.patterns)
        self.init_val_per_pattern = init_value / num_patterns if num_patterns > 0 else 0.0
        # 利用 defaultdict 與頂層 ConstantFactory 來指定預設初始值
        self.weights = [defaultdict(ConstantFactory(self.init_val_per_pattern)) for _ in patterns]
        
        if self.use_tc:
            self.tc_E = [defaultdict(float) for _ in patterns]
            self.tc_A = [defaultdict(float) for _ in patterns]
        
        # 產生所有對稱變換
        self.symmetry_patterns = []
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            for sym in syms:
                self.symmetry_patterns.append(sym)
        self.sym_per_pattern = 8 if num_patterns > 0 else 0

    def generate_symmetries(self, pattern):
        board_size = self.board_size
        transform_funcs = [identity, rot90, rot180, rot270, reflect_horizontal, reflect_vertical, reflect_main, reflect_anti]
        sym_patterns = []
        for tf in transform_funcs:
            new_pattern = []
            for idx in pattern:
                r = idx // board_size
                c = idx % board_size
                new_r, new_c = tf((r, c), board_size)
                new_idx = new_r * board_size + new_c
                new_pattern.append(new_idx)
            if new_pattern not in sym_patterns:
                sym_patterns.append(new_pattern)
        return sym_patterns

    def tile_to_index(self, tile):
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords):
        if isinstance(board, np.ndarray) and board.ndim > 1:
            flat_board = board.flatten()
        else:
            flat_board = board
        return tuple(self.tile_to_index(flat_board[i]) for i in coords)

    def value(self, board):
        total_value = 0.0
        num_patterns = len(self.patterns)
        for i in range(num_patterns):
            sym_group = self.symmetry_patterns[i * self.sym_per_pattern:(i + 1) * self.sym_per_pattern]
            pattern_value = 0.0
            for pat in sym_group:
                feature = self.get_feature(board, pat)
                pattern_value += self.weights[i][feature]
            pattern_value /= len(sym_group)
            total_value += pattern_value
        return total_value

    def update(self, board, delta, alpha):
        num_patterns = len(self.patterns)
        for i in range(num_patterns):
            sym_group = self.symmetry_patterns[i * self.sym_per_pattern:(i + 1) * self.sym_per_pattern]
            num_sym = len(sym_group)
            for pat in sym_group:
                feature = self.get_feature(board, pat)
                if self.use_tc:
                    self.tc_E[i][feature] += delta
                    self.tc_A[i][feature] += abs(delta)
                    beta = abs(self.tc_E[i][feature]) / self.tc_A[i][feature] if self.tc_A[i][feature] != 0 else 1.0
                    self.weights[i][feature] += alpha * beta * (delta / num_sym)
                else:
                    self.weights[i][feature] += alpha * (delta / num_sym)
