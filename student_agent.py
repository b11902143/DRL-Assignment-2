import os
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import math

# ----------------- 利用 gdown 從 Google Drive 下載 value.pkl -----------------
try:
    import gdown
except ImportError:
    raise ImportError("請先使用 pip install gdown 安裝 gdown 模組")

VALUE_PKL = "value.pkl"
if not os.path.exists(VALUE_PKL):
    # Google Drive 連結: https://drive.google.com/file/d/1t9i3fp1DKTsrUuaYAAu7Swrv6NovqIsZ/view?usp=drive_link
    # 轉換成下載連結：https://drive.google.com/uc?id=1t9i3fp1DKTsrUuaYAAu7Swrv6NovqIsZ
    file_id = "1t9i3fp1DKTsrUuaYAAu7Swrv6NovqIsZ"
    url = f"https://drive.google.com/uc?id={file_id}"
    print("Downloading value.pkl from Google Drive ...")
    gdown.download(url, VALUE_PKL, quiet=False)

# ----------------- Environment 定義 -----------------
COLOR_MAP = {
    0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
    16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72",
    256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e",
    4096: "#3c3a32", 8192: "#3c3a32", 16384: "#3c3a32", 32768: "#3c3a32"
}
TEXT_COLOR = {
    2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2",
    32: "#f9f6f2", 64: "#f9f6f2", 128: "#f9f6f2", 256: "#f9f6f2",
    512: "#f9f6f2", 1024: "#f9f6f2", 2048: "#f9f6f2", 4096: "#f9f6f2"
}

class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()
        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True  # Record if last move was valid

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
        new_row = row[row != 0]  # Remove zeros
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
        """Check if the specified move is legal."""
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
# ----------------- NTuple Approximator 定義 -----------------
class NTupleApproximator:
    def __init__(self, board_size, patterns):
        self.board_size = board_size
        self.patterns = patterns
        # 為每個模式建立一個權重字典
        self.weights = [dict() for _ in patterns]
        # 為每組模式產生 8 個對稱變換結果，依序儲存在 symmetry_patterns 中
        self.symmetry_patterns = []
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            for sym in syms:
                self.symmetry_patterns.append(sym)
        # 假設每組模式產生 8 個對稱結果
        self.sym_per_pattern = 8 if len(self.patterns) > 0 else 0

    def generate_symmetries(self, pattern):
        board_size = self.board_size
        transform_funcs = [
            lambda coord: coord,  # identity
            lambda coord: (coord[1], board_size - 1 - coord[0]),      # 90° rotation
            lambda coord: (board_size - 1 - coord[0], board_size - 1 - coord[1]),  # 180° rotation
            lambda coord: (board_size - 1 - coord[1], coord[0]),      # 270° rotation
            lambda coord: (coord[0], board_size - 1 - coord[1]),      # horizontal reflection
            lambda coord: (board_size - 1 - coord[0], coord[1]),      # vertical reflection
            lambda coord: (coord[1], coord[0]),                       # main diagonal reflection
            lambda coord: (board_size - 1 - coord[1], board_size - 1 - coord[0])  # anti-diagonal reflection
        ]
        sym_patterns = []
        for tf in transform_funcs:
            new_pattern = []
            for idx in pattern:
                r, c = idx // board_size, idx % board_size
                new_r, new_c = tf((r, c))
                new_pattern.append(new_r * board_size + new_c)
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
            flat = board.flatten()
        else:
            flat = board
        return tuple(self.tile_to_index(flat[i]) for i in coords)

    def value(self, board):
        total = 0.0
        num_patterns = len(self.patterns)
        for i in range(num_patterns):
            sym_group = self.symmetry_patterns[i * self.sym_per_pattern:(i+1)*self.sym_per_pattern]
            group_val = 0.0
            for pat in sym_group:
                feature = self.get_feature(board, pat)
                w = self.weights[i].get(feature, 0.0)
                group_val += w
            group_val /= len(sym_group)
            total += group_val
        return total

    def update(self, board, delta, alpha):
        num_patterns = len(self.patterns)
        for i in range(num_patterns):
            sym_group = self.symmetry_patterns[i * self.sym_per_pattern:(i+1)*self.sym_per_pattern]
            for pat in sym_group:
                feature = self.get_feature(board, pat)
                if feature not in self.weights[i]:
                    self.weights[i][feature] = 0.0
                self.weights[i][feature] += alpha * (delta / len(sym_group))

# 定義固定的 n-tuple 模式 (與訓練時設定相同)
TUPLES = [
    [0, 1, 2, 4, 5, 6],
    [1, 2, 5, 6, 9, 13],
    [0, 1, 2, 3, 4, 5],
    [0, 1, 5, 6, 7, 10],
    [0, 1, 2, 5, 9, 10],
    [0, 1, 5, 9, 13, 14],
    [0, 1, 5, 8, 9, 13],
    [0, 1, 2, 4, 6, 10],
]

# ================= simulate_move 函式 =================
def simulate_move(board, action, env):
    """Simulate the board state after an action (without adding a random tile)."""
    env = copy.deepcopy(env)
    board_copy = board.copy()
    size = env.size
    if action == 0:  # Up
        for j in range(size):
            col = board_copy[:, j]
            new_col = env.compress(col)
            new_col = env.merge(new_col)
            new_col = env.compress(new_col)
            board_copy[:, j] = new_col
    elif action == 1:  # Down
        for j in range(size):
            col = board_copy[:, j][::-1].copy()
            new_col = env.compress(col)
            new_col = env.merge(new_col)
            new_col = env.compress(new_col)
            board_copy[:, j] = new_col[::-1]
    elif action == 2:  # Left
        for i in range(size):
            row = board_copy[i]
            new_row = env.compress(row)
            new_row = env.merge(new_row)
            new_row = env.compress(new_row)
            board_copy[i] = new_row
    elif action == 3:  # Right
        for i in range(size):
            row = board_copy[i][::-1].copy()
            new_row = env.compress(row)
            new_row = env.merge(new_row)
            new_row = env.compress(new_row)
            board_copy[i] = new_row[::-1]
    return board_copy

def load_weights(approximator, filepath):
    with open(filepath, "rb") as f:
        approximator.weights = pickle.load(f)
    print(f"Weights loaded from: {filepath}")
    return approximator.weights

# ----------------- TD-MCTS 節點與 TD-MCTS 定義 -----------------
class TD_MCTS_Node:
    def __init__(self, state, score, env, parent=None, action=None):
        """
        state: 當前棋盤狀態 (numpy array)
        score: 當前得分
        env: 遊戲環境 (用來判斷該狀態下哪些動作合法)
        parent: 父節點 (None 代表 root)
        action: 從父節點到本節點所採的動作
        """
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]
    
    def fully_expanded(self):
        return len(self.untried_actions) == 0

# ---------------- TD-MCTS 類別 ----------------
class TD_MCTS:
    def __init__(self, env, approximator, iterations=100, exploration_constant=1.41, rollout_depth=0, gamma=0.99):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def create_env_from_state(self, state, score):
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        Qs = [child.total_reward / child.visits for child in node.children.values()]
        Q_min, Q_max = min(Qs), max(Qs)
        span = Q_max - Q_min if Q_max > Q_min else 1.0

        best_score = -float('inf')
        best_child = None
        for child in node.children.values():
            Q_raw = child.total_reward / child.visits
            Q_norm = (Q_raw - Q_min) / span
            U = self.c * math.sqrt(math.log(node.visits) / child.visits)
            score = Q_norm + U
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def rollout(self, sim_env, depth):
        sim_env.score = 0
        total_reward = 0.0
        discount = 1.0
        prev_score = 0
        
        for _ in range(depth):
            legal = [a for a in range(4) if sim_env.is_move_legal(a)]
            if not legal:
                break
            a = random.choice(legal)
            _, new_score, done, _ = sim_env.step(a)
            r = new_score - prev_score
            total_reward += discount * r
            discount *= self.gamma
            prev_score = new_score
            if done:
                break

        # 採用 one‑step lookahead 以 approximator 評估
        leaf_legal = [a for a in range(4) if sim_env.is_move_legal(a)]
        if leaf_legal:
            best_leaf = -float('inf')
            for a in leaf_legal:
                sim_board = simulate_move(sim_env.board, a, sim_env)
                val = self.approximator.value(sim_board)
                if val > best_leaf:
                    best_leaf = val
            total_reward += discount * best_leaf
        return total_reward

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)
        # 選擇階段
        while node.fully_expanded() and node.children:
            node = self.select_child(node)
            sim_env.board = simulate_move(sim_env.board, node.action, sim_env)
            sim_env.add_random_tile()

        # 展開階段
        if node.untried_actions:
            a = random.choice(node.untried_actions)
            sim_env.board = simulate_move(sim_env.board, a, sim_env)
            sim_env.add_random_tile()
            child = TD_MCTS_Node(state=sim_env.board.copy(), score=sim_env.score, env=self.env, parent=node, action=a)
            node.children[a] = child
            node.untried_actions.remove(a)
            node = child

        reward = self.rollout(sim_env, self.rollout_depth)
        self.backpropagate(node, reward)

    def best_action_distribution(self, root):
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        distribution = np.zeros(4)
        if best_action is not None:
            distribution[best_action] = 1.0
        return best_action, distribution

# ----------------- Agent 決策函式 -----------------
def get_action(state, score):
    # 根據傳入 state 與 score 建立新的環境實例
    env = Game2048Env()
    env.board = state.copy()
    env.score = score

    global approximator, mcts
    try:
        approximator
    except NameError:
        approximator = NTupleApproximator(board_size=4, patterns=TUPLES)
        load_weights(approximator, VALUE_PKL)
    mcts = TD_MCTS(env, approximator, iterations=100, exploration_constant=1.41, rollout_depth=10, gamma=0.99)
    root = TD_MCTS_Node(env.board.copy(), env.score, env)
    for _ in range(mcts.iterations):
        mcts.run_simulation(root)
    best_act, _ = mcts.best_action_distribution(root)
    if best_act is None:
        legal = [a for a in range(4) if env.is_move_legal(a)]
        best_act = random.choice(legal) if legal else random.choice(range(4))
    return best_act

# ----------------- __main__ 測試區 (選做) -----------------
if __name__ == "__main__":
    env = Game2048Env()
    approximator = NTupleApproximator(board_size=4, patterns=TUPLES)
    load_weights(approximator, VALUE_PKL)
    mcts = TD_MCTS(env, approximator, iterations=100, exploration_constant=1.41, rollout_depth=0, gamma=0.99)

    state = env.reset()
    env.render()

    done = False
    while not done:
        root = TD_MCTS_Node(state, env.score, env)
        for _ in range(mcts.iterations):
            mcts.run_simulation(root)
        action = mcts.best_action_distribution(root)[0]
        print("Selected action:", action)
        state, reward, done, _ = env.step(action)
        env.render(action=action)
    print("Game over, final score:", env.score)
