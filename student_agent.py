import copy
import random
import math
import numpy as np
import pickle
import gym
from gym import spaces
import matplotlib.pyplot as plt
from collections import defaultdict

# -------------------------------
# Game2048Env 定義
# -------------------------------
# 這裡保留環境類別，方便在推論時使用 (例如模擬移動)
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
        # 檢查橫向
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False
        # 檢查縱向
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


# -------------------------------
# NTuple Approximator 以及相關對稱變換函數定義
# -------------------------------
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
    def __init__(self, board_size, patterns):
        """
        初始化 N-Tuple approximator
        :param board_size: 棋盤大小 (例如 4)
        :param patterns: 每個 n-tuple 模式 (以棋盤位置 index 的 list 表示)
        """
        self.board_size = board_size
        self.patterns = patterns
        # 為每組模式建立權重字典 (同組所有對稱共享)
        self.weights = [defaultdict(float) for _ in patterns]
        # 產生所有對稱模式
        self.symmetry_patterns = []
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            for sym in syms:
                self.symmetry_patterns.append(sym)
        self.sym_per_pattern = 8 if len(self.patterns) > 0 else 0

    def generate_symmetries(self, pattern):
        """產生傳入模式的 8 種對稱變換，傳回不重複的結果列表"""
        board_size = self.board_size
        transform_funcs = [identity, rot90, rot180, rot270,
                           reflect_horizontal, reflect_vertical, reflect_main, reflect_anti]
        sym_patterns = []
        for tf in transform_funcs:
            new_pattern = []
            for idx in pattern:
                r = idx // board_size
                c = idx % board_size
                new_r, new_c = tf((r, c), board_size)
                new_pattern.append(new_r * board_size + new_c)
            if new_pattern not in sym_patterns:
                sym_patterns.append(new_pattern)
        return sym_patterns

    def tile_to_index(self, tile):
        """將 tile 數值轉換成查表索引，0 -> 0; 非 0 則取 log2(tile)"""
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords):
        """
        根據給定 coordinates 從棋盤中提取 tile 值轉換成特徵 tuple
        :param board: 棋盤 (若是 2D numpy 陣列則先扁平化)
        :param coords: 位置 index 的 list
        :return: tuple 形式的特徵
        """
        if isinstance(board, np.ndarray) and board.ndim > 1:
            flat_board = board.flatten()
        else:
            flat_board = board
        return tuple(self.tile_to_index(flat_board[i]) for i in coords)

    def value(self, board):
        """
        估計棋盤狀態的價值，對每組原始模式取所有對稱值平均後累加
        """
        total_value = 0.0
        num_patterns = len(self.patterns)
        for i in range(num_patterns):
            sym_group = self.symmetry_patterns[i * self.sym_per_pattern : (i + 1) * self.sym_per_pattern]
            pattern_value = 0.0
            for pat in sym_group:
                feature = self.get_feature(board, pat)
                pattern_value += self.weights[i][feature]
            pattern_value /= len(sym_group)
            total_value += pattern_value
        return total_value

    def update(self, board, delta, alpha):
        """
        根據 TD 誤差更新權重，對每個對稱均分更新量
        """
        num_patterns = len(self.patterns)
        for i in range(num_patterns):
            sym_group = self.symmetry_patterns[i * self.sym_per_pattern : (i + 1) * self.sym_per_pattern]
            for pat in sym_group:
                feature = self.get_feature(board, pat)
                self.weights[i][feature] += alpha * (delta / len(sym_group))


# -------------------------------
# 模擬移動函數 (用於評估每個動作)
# -------------------------------
def simulate_move(board, action, env):
    """
    根據給定動作模擬盤面移動，不包含隨機新增 tile
    :param board: 當前棋盤狀態 (numpy 陣列)
    :param action: 動作編號 (0:上, 1:下, 2:左, 3:右)
    :param env: 2048 環境 (用於使用 compress 與 merge 方法)
    :return: 模擬後的新盤面狀態 (複製後的 numpy 陣列)
    """
    board_copy = board.copy()
    size = env.size
    if action == 0:  # 上移
        for j in range(size):
            col = board_copy[:, j]
            new_col = env.compress(col)
            new_col = env.merge(new_col)
            new_col = env.compress(new_col)
            board_copy[:, j] = new_col
    elif action == 1:  # 下移
        for j in range(size):
            col = board_copy[:, j][::-1].copy()
            new_col = env.compress(col)
            new_col = env.merge(new_col)
            new_col = env.compress(new_col)
            board_copy[:, j] = new_col[::-1]
    elif action == 2:  # 左移
        for i in range(size):
            row = board_copy[i]
            new_row = env.compress(row)
            new_row = env.merge(new_row)
            new_row = env.compress(new_row)
            board_copy[i] = new_row
    elif action == 3:  # 右移
        for i in range(size):
            row = board_copy[i][::-1].copy()
            new_row = env.compress(row)
            new_row = env.merge(new_row)
            new_row = env.compress(new_row)
            board_copy[i] = new_row[::-1]
    return board_copy

# -------------------------------
# 定義 2048 常用的 n-tuple 模式 (可依需求調整)
# -------------------------------
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

# -------------------------------
# 輔助函數：檢查給定 state 下某動作是否合法
# -------------------------------
def is_move_legal_state(state, action, env):
    """
    判斷給定 state 之下，動作是否會改變盤面（即是否合法）
    :param state: 當前盤面 (numpy array)
    :param action: 動作 (0~3)
    :param env: Game2048Env 實例 (用於 compress 與 merge)
    :return: True 如果模擬後盤面有改變，否則 False
    """
    new_state = simulate_move(state, action, env)
    return not np.array_equal(new_state, state)

# -------------------------------
# 推論階段的決策函數：get_action(state, score)
# -------------------------------
# 此函數將由評分系統調用，根據當前盤面及分數選擇一個動作 (0~3)
approximator = None
env = None

def get_action(state, score):
    """
    根據輸入的 state 與 score 回傳一個動作 (0~3)。
    使用外部訓練產生的權重 (value.pkl) 及 NTupleApproximator 進行評估，
    並選出使盤面價值最高的合法動作。
    """
    global approximator, env
    # 第一次呼叫時初始化環境與 approximator 並載入訓練後權重
    if env is None or approximator is None:
        env = Game2048Env()
        approximator = NTupleApproximator(board_size=4, patterns=pattern)
        try:
            with open("value.pkl", "rb") as f:
                approximator.weights = pickle.load(f)
        except Exception as e:
            # 如果無法載入權重，則打印錯誤並回傳隨機動作
            print("Warning: 無法載入 value.pkl，將以隨機動作回應。", e)
            return random.choice(range(4))
    
    # 取得所有合法動作 (使用輔助函數判斷)
    legal_moves = [a for a in range(4) if is_move_legal_state(state, a, env)]
    if not legal_moves:
        # 若無合法動作，回傳隨機一個動作 (通常不會發生)
        return random.choice(range(4))

    best_value = -float('inf')
    best_action = legal_moves[0]
    for a in legal_moves:
        sim_state = simulate_move(state, a, env)
        value_est = approximator.value(sim_state)
        if value_est > best_value:
            best_value = value_est
            best_action = a
    return best_action

# -------------------------------
# 測試區段 (僅供本地測試用，不會於評分環境中呼叫 get_action 之外的內容)
# -------------------------------
if __name__ == "__main__":
    # 用 get_action 測試單一盤面動作決策
    test_env = Game2048Env()
    current_state = test_env.reset()
    current_score = 0
    print("初始盤面：")
    print(current_state)
    action = get_action(current_state, current_score)
    print("選擇動作：", action)
    
    # 示範執行該動作
    next_state, new_score, done, _ = test_env.step(action)
    print("執行動作後的盤面：")
    print(next_state)
