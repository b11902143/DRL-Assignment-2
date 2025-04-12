import math
import copy
import random
import pickle
import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
from collections import defaultdict

##############################################
# 注意：下面這段 Game2048Env 的程式碼完全保留了你的環境，不做任何修改
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
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
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
# 以下為 n-tuple approximator 與 TD-MCTS 相關程式碼
##############################################
# 對稱變換函式
def identity(coord, board_size):
    return coord
def rot90(coord, board_size):
    r, c = coord
    return (c, board_size-1-r)
def rot180(coord, board_size):
    r, c = coord
    return (board_size-1-r, board_size-1-c)
def rot270(coord, board_size):
    r, c = coord
    return (board_size-1-c, r)
def reflect_horizontal(coord, board_size):
    r, c = coord
    return (r, board_size-1-c)
def reflect_vertical(coord, board_size):
    r, c = coord
    return (board_size-1-r, c)
def reflect_main(coord, board_size):
    r, c = coord
    return (c, r)
def reflect_anti(coord, board_size):
    r, c = coord
    return (board_size-1-c, board_size-1-r)

# 預設的 n-tuple 模式（2048 常見配置）
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
        # 這裡用一般字典保存權重；預設權重將在取值時回傳 self.init_val_per_pattern
        self.weights = [{} for _ in patterns]
        if self.use_tc:
            self.tc_E = [{} for _ in patterns]
            self.tc_A = [{} for _ in patterns]
        self.symmetry_patterns = []
        for p in self.patterns:
            syms = self.generate_symmetries(p)
            for s in syms:
                self.symmetry_patterns.append(s)
        self.sym_per_pattern = 8 if num_patterns > 0 else 0

    def generate_symmetries(self, pattern):
        board_size = self.board_size
        funcs = [identity, rot90, rot180, rot270,
                 reflect_horizontal, reflect_vertical, reflect_main, reflect_anti]
        sym_patterns = []
        for f in funcs:
            new_p = []
            for idx in pattern:
                r = idx // board_size
                c = idx % board_size
                nr, nc = f((r, c), board_size)
                new_p.append(nr*board_size + nc)
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
            group = self.symmetry_patterns[i*self.sym_per_pattern:(i+1)*self.sym_per_pattern]
            val = 0.0
            for pat in group:
                feat = self.get_feature(board, pat)
                val += self.weights[i].get(feat, self.init_val_per_pattern)
            total += val / len(group)
        return total

    def update(self, board, delta, alpha):
        num = len(self.patterns)
        for i in range(num):
            group = self.symmetry_patterns[i*self.sym_per_pattern:(i+1)*self.sym_per_pattern]
            n_sym = len(group)
            for pat in group:
                feat = self.get_feature(board, pat)
                if self.use_tc:
                    curE = self.tc_E[i].get(feat, 0.0)
                    curA = self.tc_A[i].get(feat, 0.0)
                    self.tc_E[i][feat] = curE + delta
                    self.tc_A[i][feat] = curA + abs(delta)
                    beta = abs(self.tc_E[i][feat]) / self.tc_A[i][feat] if self.tc_A[i][feat] != 0 else 1.0
                    curr = self.weights[i].get(feat, self.init_val_per_pattern)
                    self.weights[i][feat] = curr + alpha * beta * (delta / n_sym)
                else:
                    curr = self.weights[i].get(feat, self.init_val_per_pattern)
                    self.weights[i][feat] = curr + alpha * (delta / n_sym)

##############################################
# 輔助函式：用於判斷某盤面下，指定動作是否合法
##############################################
def is_move_legal_for_state(state, action):
    temp_env = Game2048Env()
    temp_env.board = state.copy()
    return temp_env.is_move_legal(action)

##############################################
# 模擬盤面移動及計算 reward（不 spawn random tile）
##############################################
def simulate_move(board, action, env):
    board_cp = board.copy()
    size = env.size
    if action == 0:
        for j in range(size):
            col = board_cp[:, j]
            new_col = env.compress(col)
            new_col = env.merge(new_col)
            new_col = env.compress(new_col)
            board_cp[:, j] = new_col
    elif action == 1:
        for j in range(size):
            col = board_cp[:, j][::-1].copy()
            new_col = env.compress(col)
            new_col = env.merge(new_col)
            new_col = env.compress(new_col)
            board_cp[:, j] = new_col[::-1]
    elif action == 2:
        for i in range(size):
            row = board_cp[i]
            new_row = env.compress(row)
            new_row = env.merge(new_row)
            new_row = env.compress(new_row)
            board_cp[i] = new_row
    elif action == 3:
        for i in range(size):
            row = board_cp[i][::-1].copy()
            new_row = env.compress(row)
            new_row = env.merge(new_row)
            new_row = env.compress(new_row)
            board_cp[i] = new_row[::-1]
    return board_cp

def merge_and_reward(row):
    reward = 0
    row_cp = row.copy()
    for i in range(len(row_cp)-1):
        if row_cp[i]!=0 and row_cp[i]==row_cp[i+1]:
            row_cp[i]*=2
            row_cp[i+1]=0
            reward += row_cp[i]
    return row_cp, reward

def simulate_move_and_reward(board, action, env):
    board_cp = board.copy()
    reward = 0
    size = env.size
    if action == 0:
        for j in range(size):
            col = board_cp[:, j].copy()
            new_col = env.compress(col)
            new_col, r = merge_and_reward(new_col)
            new_col = env.compress(new_col)
            board_cp[:, j] = new_col
            reward += r
    elif action == 1:
        for j in range(size):
            col = board_cp[:, j][::-1].copy()
            new_col = env.compress(col)
            new_col, r = merge_and_reward(new_col)
            new_col = env.compress(new_col)
            board_cp[:, j] = new_col[::-1]
            reward += r
    elif action == 2:
        for i in range(size):
            row = board_cp[i].copy()
            new_row = env.compress(row)
            new_row, r = merge_and_reward(new_row)
            new_row = env.compress(new_row)
            board_cp[i] = new_row
            reward += r
    elif action == 3:
        for i in range(size):
            row = board_cp[i][::-1].copy()
            new_row = env.compress(row)
            new_row, r = merge_and_reward(new_row)
            new_row = env.compress(new_row)
            board_cp[i] = new_row[::-1]
            reward += r
    return board_cp, reward

##############################################
# TD-MCTS 節點定義
##############################################
class TD_MCTS_Node:
    def __init__(self, state, score, parent=None, action=None):
        self.state = state.copy()
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}  # action: child_node
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = [a for a in range(4) if is_move_legal_for_state(self.state, a)]
        
    def fully_expanded(self):
        return len(self.untried_actions) == 0

##############################################
# TD-MCTS 主類別（利用預訓練 approximator 進行葉節點評估）
##############################################
class TD_MCTS:
    def __init__(self, env, approximator, iterations=50, exploration_constant=1.41, rollout_depth=10, gamma=0.99):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def create_env_from_state(self, state, score):
        new_env = Game2048Env()
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        best_val = -float('inf')
        best_child = None
        for action, child in node.children.items():
            if child.visits == 0:
                return child
            avg = child.total_reward / child.visits
            uct = avg + self.c * math.sqrt(math.log(node.visits) / child.visits)
            if uct > best_val:
                best_val = uct
                best_child = child
        return best_child

    def rollout(self, sim_env, depth):
        discount = 1.0
        total = 0.0
        steps = 0
        while steps < depth and not sim_env.is_game_over():
            legal = [a for a in range(4) if sim_env.is_move_legal(a)]
            if not legal:
                break
            action = random.choice(legal)
            next_board, r = simulate_move_and_reward(sim_env.board, action, sim_env)
            total += discount * r
            discount *= self.gamma
            sim_env.board = next_board.copy()
            sim_env.score += r
            steps += 1
            if sim_env.is_game_over():
                break
        heuristic = self.approximator.value(sim_env.board)
        total += discount * heuristic
        return total

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            reward *= self.gamma
            node = node.parent

    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)
        while not sim_env.is_game_over() and node.fully_expanded():
            node = self.select_child(node)
            sim_env = self.create_env_from_state(node.state, node.score)
        if not sim_env.is_game_over() and node.untried_actions:
            action = random.choice(node.untried_actions)
            node.untried_actions.remove(action)
            next_board, r = simulate_move_and_reward(sim_env.board, action, sim_env)
            child = TD_MCTS_Node(state=next_board, score=sim_env.score + r, parent=node, action=action)
            node.children[action] = child
            sim_env.board = next_board.copy()
            sim_env.score += r
            node = child
        rollout_reward = self.rollout(sim_env, self.rollout_depth)
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_act = None
        best_visits = -1
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_act = action
        return best_act, distribution

##############################################
# 載入預訓練權重：讀取 value.pkl 並初始化 approximator
##############################################
# 建立 approximator（參數需與訓練時一致）
approximator = NTupleApproximator(board_size=4, patterns=pattern, init_value=320000, use_tc=True)
with open("value.pkl", "rb") as f:
    approximator.weights = pickle.load(f)

##############################################
# Agent 介面：get_action(state, score)
##############################################
def get_action(state, score):
    """
    輸入:
      state: 當前盤面 (numpy 陣列)
      score: 當前分數
    輸出:
      動作 (0～3)，利用 TD-MCTS 與預訓練 approximator 選擇
    """
    env = Game2048Env()
    env.board = state.copy()
    env.score = score
    root = TD_MCTS_Node(state=state, score=score)
    td_mcts = TD_MCTS(env=env, approximator=approximator, iterations=50, exploration_constant=1.41, rollout_depth=10, gamma=0.99)
    for _ in range(td_mcts.iterations):
        td_mcts.run_simulation(root)
    best_act, _ = td_mcts.best_action_distribution(root)
    return best_act

##############################################
# 測試區段 (僅供本地測試用；評測系統將呼叫 get_action)
##############################################
if __name__ == "__main__":
    env = Game2048Env()
    state = env.reset()
    env.render()
    done = False
    while not done:
        act = get_action(state, env.score)
        print("Selected action:", act)
        state, score, done, _ = env.step(act)
        env.render(action=act)
    print("Game over. Final score:", env.score)
