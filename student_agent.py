import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import math

### ================= Environment 定義 =================
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
        self.size = 4  # 4x4 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]
        self.last_move_valid = True
        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def merge(self, row):
        for i in range(len(row) - 1):
            if row[i] == row[i+1] and row[i] != 0:
                row[i] *= 2
                row[i+1] = 0
                self.score += row[i]
        return row

    def move_left(self):
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

### ================= NTuple Approximator 定義 =================
class NTupleApproximator:
    def __init__(self, board_size, patterns):
        self.board_size = board_size
        self.patterns = patterns
        self.weights = [dict() for _ in patterns]
        self.symmetry_patterns = []
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            for sym in syms:
                self.symmetry_patterns.append(sym)
        self.sym_per_pattern = 8 if len(self.patterns) > 0 else 0

    def generate_symmetries(self, pattern):
        board_size = self.board_size
        transform_funcs = [
            lambda coord: coord,
            lambda coord: (coord[1], board_size - 1 - coord[0]),
            lambda coord: (board_size - 1 - coord[0], board_size - 1 - coord[1]),
            lambda coord: (board_size - 1 - coord[1], coord[0]),
            lambda coord: (coord[0], board_size - 1 - coord[1]),
            lambda coord: (board_size - 1 - coord[0], coord[1]),
            lambda coord: (coord[1], coord[0]),
            lambda coord: (board_size - 1 - coord[1], board_size - 1 - coord[0])
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
            sym_group = self.symmetry_patterns[i*self.sym_per_pattern:(i+1)*self.sym_per_pattern]
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
            sym_group = self.symmetry_patterns[i*self.sym_per_pattern:(i+1)*self.sym_per_pattern]
            for pat in sym_group:
                feature = self.get_feature(board, pat)
                if feature not in self.weights[i]:
                    self.weights[i][feature] = 0.0
                self.weights[i][feature] += alpha * (delta / len(sym_group))

# 定義 n-tuple 模式 (2048 常見模式)
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

### ================= simulate_move 函式 =================
def simulate_move(board, action, env):
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

### ================= TD-MCTS Node 與 TD-MCTS 定義 =================
class TD_MCTS_Node:
    def __init__(self, state, score, parent=None, action=None):
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        # children: 若同一玩家動作會產生多個 outcome，儲存為 { action: { (pos, tile): child, ... } }
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]
    
    def fully_expanded(self):
        return len(self.untried_actions) == 0

class TD_MCTS:
    def __init__(self, env, approximator, iterations=50, exploration_constant=1.41, rollout_depth=10, gamma=0.99):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def is_terminal(self, sim_env):
        legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
        return len(legal_moves) == 0

    def create_env_from_state(self, state, score):
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        best_child = None
        best_uct = -float('inf')
        for action, child in node.children.items():
            if isinstance(child, dict):
                representative = None
                max_visits = -1
                for outcome_key, chance_child in child.items():
                    if chance_child.visits > max_visits:
                        max_visits = chance_child.visits
                        representative = chance_child
                current_child = representative
            else:
                current_child = child
            exploitation = current_child.total_reward / current_child.visits if current_child.visits > 0 else 0
            exploration = self.c * math.sqrt(math.log(node.visits) / current_child.visits) if current_child.visits > 0 else float('inf')
            exploitation /= 1000  # 若需要可以調整 scale
            uct_value = exploitation + exploration
            if uct_value > best_uct:
                best_uct = uct_value
                best_child = current_child
        return best_child

    def rollout(self, sim_env, depth):
        total_reward = 0.0
        discount = 1.0
        current_depth = 0
        while current_depth < depth:
            legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
            if not legal_moves:
                break
            action = random.choice(legal_moves)
            new_state, reward, done, _ = sim_env.step(action)
            total_reward += discount * reward
            discount *= self.gamma
            current_depth += 1
            if done:
                break
        value_est = self.approximator.value(sim_env.board)
        total_reward += discount * value_est
        return total_reward

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            reward *= self.gamma
            node = node.parent

    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)
        
        while node.fully_expanded() and not self.is_terminal(sim_env):
            node = self.select_child(node)
            sim_env = self.create_env_from_state(node.state, node.score)
        
        if not self.is_terminal(sim_env) and node.untried_actions:
            action = random.choice(node.untried_actions)
            node.untried_actions.remove(action)
            afterstate = simulate_move(sim_env.board, action, sim_env)
            empty_positions = list(zip(*np.where(afterstate == 0)))
            if empty_positions:
                chance_children = {}
                for pos in empty_positions:
                    for tile_value in [2, 4]:
                        new_state = afterstate.copy()
                        r, c = pos
                        new_state[r, c] = tile_value
                        child_node = TD_MCTS_Node(new_state, sim_env.score, parent=node, action=action)
                        chance_children[(pos, tile_value)] = child_node
                node.children[action] = chance_children
                if chance_children:
                    chosen_key = random.choice(list(chance_children.keys()))
                    node = chance_children[chosen_key]
                    sim_env = self.create_env_from_state(node.state, node.score)
                else:
                    child_node = TD_MCTS_Node(afterstate, sim_env.score, parent=node, action=action)
                    node.children[action] = child_node
                    node = child_node
                    sim_env = self.create_env_from_state(node.state, node.score)
            else:
                child_node = TD_MCTS_Node(afterstate, sim_env.score, parent=node, action=action)
                node.children[action] = child_node
                node = child_node
                sim_env = self.create_env_from_state(node.state, node.score)
        
        rollout_reward = self.rollout(sim_env, self.rollout_depth)
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        total_visits = 0
        for action, child in root.children.items():
            if isinstance(child, dict):
                total_visits += sum(c.visits for c in child.values())
            else:
                total_visits += child.visits
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            if isinstance(child, dict):
                action_visits = sum(c.visits for c in child.values())
            else:
                action_visits = child.visits
            distribution[action] = action_visits / total_visits if total_visits > 0 else 0
            if action_visits > best_visits:
                best_visits = action_visits
                best_action = action
        return best_action, distribution

### ================= 載入 value.pkl 的 approximator 權重 =================
# 全域建立環境與 approximator
env = Game2048Env()
approximator = NTupleApproximator(board_size=4, patterns=TUPLES)
with open("value.pkl", "rb") as f:
    approximator.weights = pickle.load(f)
print("Loaded approximator weights from value.pkl")

### ================= get_action 函式 =================
def get_action(state, score):
    """
    輸入: state (numpy array of board), score (int)
    輸出: action (0~3)
    
    本函式根據當前狀態利用 TD-MCTS 搭配已訓練好的 N-Tuple Approximator 進行搜尋，
    並返回最佳動作。
    """
    # 建立一個環境副本並設置其 board 與 score
    current_env = copy.deepcopy(env)
    current_env.board = state.copy()
    current_env.score = score
    
    # 建立 MCTS 樹根
    root = TD_MCTS_Node(state, score)
    
    # 建立 TD-MCTS 搜尋器（可調整 iterations 與 rollout_depth）
    td_mcts = TD_MCTS(current_env, approximator, iterations=50, exploration_constant=1.41, rollout_depth=10, gamma=0.99)
    
    # 執行多次模擬以構建樹
    for _ in range(td_mcts.iterations):
        td_mcts.run_simulation(root)
    
    # 根據訪問次數選擇最佳動作
    best_act, dist = td_mcts.best_action_distribution(root)
    return best_act
