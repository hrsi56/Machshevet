# =================================================================
# peg_solitaire_ultimate.py
#
# שילוב אולטימטיבי:
# 1. ארכיטקטורת GNN (מגרסה 1)
# 2. פונקציית פוטנציאל מורכבת (PBRS) ורכיבי משחק (מגרסה 2)
# 3. סוכן אימון מתקדם עם PER ו-MCTS משופר (שילוב של שניהם)
# 4. GUI אינטראקטיבי (מגרסה 2)
#
# להרצה, יש להתקין את הספריות הנדרשות (מומלץ בסביבה וירטואלית):
# pip install torch numpy tqdm
# pip install torch_geometric
# pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-$(python -c 'import torch; print(torch.__version__)').html
# =================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
import copy
import random
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional, Callable, Union
from collections import deque
import tkinter as tk
from tkinter import messagebox
import heapq
import itertools
from pathlib import Path
import pickle


# =================================================================
# 1. מחלקות המשחק והלוח (שילוב והתאמה)
# =================================================================

class Board:
	"""מייצג את לוח הסוליטר. 1 לפיון, 0 לחור."""
	LEGAL_POSITIONS = sorted([
		(r, c) for r in range(7) for c in range(7)
		if not ((r < 2 or r > 4) and (c < 2 or c > 4))
	])
	TOTAL_PEGS = len(LEGAL_POSITIONS) - 1

	def __init__(self, state_dict: Optional[Dict[Tuple[int, int], int]] = None, empty_pos: Tuple[int, int] = (3, 3)):
		if state_dict:
			self.state = state_dict.copy()
		else:
			self.state = {pos: 1 for pos in self.LEGAL_POSITIONS}
			if empty_pos in self.state:
				self.state[empty_pos] = 0

	def get(self, pos: Tuple[int, int]) -> Optional[int]:
		return self.state.get(pos)

	def set(self, pos: Tuple[int, int], value: int):
		self.state[pos] = value

	def count_pegs(self) -> int:
		return sum(self.state.values())

	def as_array(self) -> np.ndarray:
		arr = np.zeros((7, 7), dtype=np.float32)
		for p, v in self.state.items():
			arr[p] = float(v)
		return arr

	def copy(self) -> "Board":
		return Board(state_dict=self.state)


class PegActionSpace:
	"""מנהל את מרחב הפעולות האפשריות במשחק."""

	def __init__(self, board_size: int = 7):
		self.board_size = board_size
		self.directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]  # Up, Down, Left, Right
		self.num_actions = self.board_size * self.board_size * len(self.directions)
		self._action_to_index: Dict[Tuple[int, int, int], int] = {}
		self._index_to_action: List[Optional[Tuple[int, int, int]]] = [None] * self.num_actions
		idx = 0
		for r in range(board_size):
			for c in range(board_size):
				for dir_idx in range(len(self.directions)):
					action = (r, c, dir_idx)
					if (r, c) in Board.LEGAL_POSITIONS:
						self._action_to_index[action] = idx
						self._index_to_action[idx] = action
					idx += 1

	def to_index(self, action: Tuple[int, int, int]) -> int:
		return self._action_to_index[action]

	def from_index(self, index: int) -> Tuple[int, int, int]:
		return self._index_to_action[index]

	def __len__(self) -> int:
		return self.num_actions


class Game:
	"""מנוע המשחק המרכזי."""
	DIRECTIONS = [(-2, 0), (2, 0), (0, -2), (0, 2)]
	ACTION_SPACE = PegActionSpace()

	def __init__(self, board: Optional[Board] = None):
		self.board = board.copy() if board else Board()
		self.move_history: List[Tuple[Tuple, Tuple, Tuple, Board]] = []
		self.redo_stack: List[Tuple[Tuple, Tuple, Tuple, Board]] = []

	def get_legal_moves(self) -> List[Tuple[int, int, int]]:
		moves = []
		for r_from, c_from in self.board.state:
			if self.board.get((r_from, c_from)) == 1:
				for dir_idx, (dr, dc) in enumerate(self.DIRECTIONS):
					to_pos = (r_from + dr, c_from + dc)
					over_pos = (r_from + dr // 2, c_from + dc // 2)
					if self.board.get(to_pos) == 0 and self.board.get(over_pos) == 1:
						moves.append((r_from, c_from, dir_idx))
		return moves

	def apply_action(self, action: Tuple[int, int, int]):
		before_state = self.board.copy()
		from_r, from_c, dir_idx = action
		dr, dc = self.DIRECTIONS[dir_idx]
		from_pos, to_pos = (from_r, from_c), (from_r + dr, from_c + dc)
		over_pos = (from_r + dr // 2, from_c + dc // 2)

		self.board.set(from_pos, 0)
		self.board.set(over_pos, 0)
		self.board.set(to_pos, 1)
		self.move_history.append((from_pos, to_pos, over_pos, before_state))
		self.redo_stack.clear()

	def undo(self) -> bool:
		if not self.move_history: return False
		last_move = self.move_history.pop()
		self.redo_stack.append(last_move)
		self.board = last_move[3]
		return True

	def redo(self) -> bool:
		if not self.redo_stack: return False
		move_to_redo = self.redo_stack.pop()
		self.apply_action((move_to_redo[0][0], move_to_redo[0][1], self.DIRECTIONS.index(
			(move_to_redo[1][0] - move_to_redo[0][0], move_to_redo[1][1] - move_to_redo[0][1]))))
		return True

	def reset(self):
		self.board = Board()
		self.move_history.clear()
		self.redo_stack.clear()

	def is_game_over(self) -> bool:
		return len(self.get_legal_moves()) == 0

	def is_win(self) -> bool:
		return self.board.count_pegs() == 1 and self.board.get((3, 3)) == 1

	def clone(self) -> "Game":
		return copy.deepcopy(self)


# =================================================================
# 2. רכיבי GNN וקידוד גרפי (שילוב ושיפור)
# =================================================================
class GraphEncoder:
	"""ממיר את לוח המשחק לייצוג גרפי."""

	def __init__(self):
		self.node_map = {pos: i for i, pos in enumerate(Board.LEGAL_POSITIONS)}
		self.num_nodes = len(self.node_map)
		# שיפור: קשתות מייצגות מהלכי קפיצה אפשריים, לא רק שכנות
		self.edge_index = self._create_jump_based_edge_index()

	def _create_jump_based_edge_index(self):
		edges = []
		for r_from, c_from in self.node_map:
			for dr, dc in Game.DIRECTIONS:
				r_to, c_to = r_from + dr, c_from + dc
				if (r_to, c_to) in self.node_map:
					edges.append([self.node_map[(r_from, c_from)], self.node_map[(r_to, c_to)]])
		return torch.tensor(edges, dtype=torch.long).t().contiguous()

	def encode(self, board: Board, move_count: int) -> Data:
		node_features = np.zeros((self.num_nodes, 4), dtype=np.float32)
		for pos, i in self.node_map.items():
			node_features[i] = [
				board.get(pos) == 1,
				pos == (3, 3),
				pos[0] / 6.0,
				pos[1] / 6.0
			]
		pegs = float(board.count_pegs())
		return Data(x=torch.from_numpy(node_features), edge_index=self.edge_index,
		            pegs_left=torch.tensor([[pegs / Board.TOTAL_PEGS]], dtype=torch.float32),
		            move_count=torch.tensor([[move_count / Board.TOTAL_PEGS]], dtype=torch.float32))


class PegSolitaireGNN(nn.Module):
	"""ארכיטקטורת רשת הגרפים."""

	def __init__(self, node_features=4, num_actions=Game.ACTION_SPACE.num_actions, gnn_channels=64, num_heads=4):
		super().__init__()
		self.conv1 = GATv2Conv(node_features, gnn_channels, heads=num_heads, concat=True, dropout=0.2)
		self.conv2 = GATv2Conv(gnn_channels * num_heads, gnn_channels, heads=1, concat=False, dropout=0.2)
		self.value_fc1 = nn.Linear(gnn_channels + 2, 64)
		self.value_fc2 = nn.Linear(64, 1)
		self.policy_fc1 = nn.Linear(gnn_channels + 2, 64)
		self.policy_fc2 = nn.Linear(64, num_actions)

	def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
		x = F.elu(self.conv1(data.x, data.edge_index))
		x = F.elu(self.conv2(x, data.edge_index))
		graph_emb = global_mean_pool(x, data.batch)
		combined = torch.cat([graph_emb, data.pegs_left, data.move_count], dim=1)
		v = torch.tanh(self.value_fc2(F.relu(self.value_fc1(combined))))
		p = self.policy_fc2(F.relu(self.policy_fc1(combined)))
		return p, v


# =================================================================
# 3. פונקציית פוטנציאל (PBRS) (מגרסה 2)
# =================================================================
class PotentialCalculator:
	"""מחשב את הפוטנציאל (Φ) של מצב לוח נתון."""
	CENTRALITY_WEIGHTS = np.array([[0, 0, .1, .1, .1, 0, 0], [0, .1, .2, .3, .2, .1, 0], [.1, .2, .4, .5, .4, .2, .1],
	                               [.1, .3, .5, 1, .5, .3, .1], [.1, .2, .4, .5, .4, .2, .1],
	                               [0, .1, .2, .3, .2, .1, 0], [0, 0, .1, .1, .1, 0, 0]], dtype=np.float32)
	PAGODA_VALUES = np.array(
		[[0, 0, 1, 2, 1, 0, 0], [0, 1, 2, 3, 2, 1, 0], [1, 2, 3, 4, 3, 2, 1], [2, 3, 4, 5, 4, 3, 2],
		 [1, 2, 3, 4, 3, 2, 1], [0, 1, 2, 3, 2, 1, 0], [0, 0, 1, 2, 1, 0, 0]], dtype=np.float32)
	CORNER_POSITIONS = [(0, 3), (3, 0), (3, 6), (6, 3)]
	EDGE_MASK = ((CENTRALITY_WEIGHTS < 0.25) & (CENTRALITY_WEIGHTS > 0)).astype(np.float32)
	K_REACH = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int8)

	@staticmethod
	def _count_truly_isolated_pegs(board: Board) -> int:
		arr = board.as_array()
		count = 0
		for r_from, c_from in board.state:
			if arr[r_from, c_from] != 1: continue
			is_isolated = True
			for dr, dc in Game.DIRECTIONS:
				r_to, c_to = r_from + dr, c_from + dc
				r_over, c_over = r_from + dr // 2, c_from + dc // 2
				if (r_to, c_to) in board.state and arr[r_over, c_over] == 1 and arr[r_to, c_to] == 0:
					is_isolated = False;
					break
			if is_isolated: count += 1
		return count

	def calculate(self, board: Board) -> float:
		from scipy.ndimage import convolve
		arr = board.as_array()
		mask = np.zeros_like(arr);
		mask[tuple(zip(*Board.LEGAL_POSITIONS))] = 1
		pegs = arr * mask
		N = pegs.sum()
		if N == 0: return -20.0

		phi0 = -N
		phi1 = (pegs * self.CENTRALITY_WEIGHTS).sum() / N if N > 0 else 0
		phi2 = (pegs * self.PAGODA_VALUES).sum()

		neigh = convolve(pegs, self.K_REACH, mode="constant", cval=0)
		reachable = (neigh > 0) & (convolve(mask - pegs, self.K_REACH, mode="constant", cval=0) > 0)
		isolated = ((pegs == 1) & (~reachable)).sum()
		corners = sum(pegs[r, c] for r, c in self.CORNER_POSITIONS)
		edges = (pegs * self.EDGE_MASK).sum()
		phi3 = -(5.0 * isolated + 2.0 * corners + 1.0 * edges)

		phi4 = -10.0 * self._count_truly_isolated_pegs(board)

		# Z-score normalization
		μ = np.array([-17.0, 0.25, 28.0, -12.0, -5.0], dtype=np.float32)
		σ = np.array([10.0, 0.12, 20.0, 8.0, 5.0], dtype=np.float32)
		φ = np.array([phi0, phi1, phi2, phi3, phi4], dtype=np.float32)
		φn = (φ - μ) / (σ + 1e-6)
		w = np.array([0.9, 0.9, 0.6, 0.8, 1.0], dtype=np.float32)
		return float((w * φn).sum())


# =================================================================
# 4. סביבה, MCTS ורכיבי אימון (שילוב ושיפור)
# =================================================================
class PegSolitaireEnv:
	"""סביבת למידת החיזוק."""

	def __init__(self, graph_encoder: GraphEncoder):
		self.graph_encoder = graph_encoder
		self.game = Game()
		self.done = False
		self.move_count = 0

	def reset(self, start_board: Optional[Board] = None):
		self.game = Game(board=start_board)
		self.done = False
		self.move_count = 0
		return self.get_observation()

	def step(self, action: Tuple[int, int, int]):
		self.game.apply_action(action)
		self.move_count += 1
		self.done = self.game.is_game_over()
		return self.get_observation(), 1.0 if self.game.is_win() else 0.0, self.done, {}

	def get_legal_moves(self) -> List[Tuple[int, int, int]]: return self.game.get_legal_moves()

	def get_observation(self) -> Data: return self.graph_encoder.encode(self.game.board, self.move_count)

	def get_game_score(self) -> float: return float(self.game.board.count_pegs())

	def clone(self) -> "PegSolitaireEnv": return copy.deepcopy(self)


class _Node:
	"""צומת פנימי בעץ החיפוש MCTS."""

	def __init__(self, prior: float):
		self.visit_count = 0
		self.value_sum = 0.0
		self.prior = prior
		self.children: Dict[int, _Node] = {}

	@property
	def value(self) -> float:
		return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0


class MCTS:
	"""אלגוריתם חיפוש עץ מונטה קרלו עם הנחיית פוטנציאל."""

	def __init__(self, model: PegSolitaireGNN, potential_calculator: PotentialCalculator, config: Dict):
		self.model = model
		self.potential_calculator = potential_calculator
		self.config = config
		self.device = config['device']
		self.action_space = Game.ACTION_SPACE

	def run(self, env: PegSolitaireEnv, temperature: float = 1.0) -> np.ndarray:
		root = _Node(0.0)
		self._expand(root, env)
		if not root.children: return np.zeros(len(self.action_space), dtype=np.float32)

		for _ in range(self.config['mcts_simulations']):
			game_copy = env.clone()
			node, path = root, [root]
			while node.children:
				action_idx, node = self._select_child(node)
				_, _, done, _ = game_copy.step(self.action_space.from_index(action_idx))
				path.append(node)

			value = self._evaluate_leaf(game_copy) if not done else (1.0 if game_copy.game.is_win() else -1.0)
			if not done and game_copy.get_legal_moves():
				self._expand(node, game_copy)

			for node_in_path in reversed(path):
				node_in_path.visit_count += 1
				node_in_path.value_sum += value

		return self._calculate_final_policy(root, temperature)

	def _select_child(self, node: _Node) -> Tuple[int, _Node]:
		sqrt_total = np.sqrt(node.visit_count)
		best_s, best_a, best_c = -np.inf, -1, None
		for a_idx, child in node.children.items():
			score = child.value + self.config['c_puct'] * child.prior * sqrt_total / (1 + child.visit_count)
			if score > best_s: best_s, best_a, best_c = score, a_idx, child
		return best_a, best_c

	def _expand(self, node: _Node, env: PegSolitaireEnv):
		legal_actions = env.get_legal_moves()
		if not legal_actions: return

		obs = env.get_observation().to(self.device)
		with torch.no_grad():
			logits, _ = self.model(obs)
			probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

		if self.config.get('dirichlet_alpha', 0) > 0 and len(legal_actions) > 1:
			noise = np.random.dirichlet([self.config['dirichlet_alpha']] * len(legal_actions))
			frac = self.config.get('noise_fraction', 0.25)
			for i, action in enumerate(legal_actions):
				idx = self.action_space.to_index(action)
				probs[idx] = probs[idx] * (1 - frac) + frac * noise[i]

		for action in legal_actions:
			idx = self.action_space.to_index(action)
			node.children[idx] = _Node(probs[idx])

	def _evaluate_leaf(self, env: PegSolitaireEnv) -> float:
		obs = env.get_observation().to(self.device)
		with torch.no_grad():
			_, net_value = self.model(obs)

		potential_value = self.potential_calculator.calculate(env.game.board)
		# שילוב ערך הרשת עם ערך הפוטנציאל
		combined_value = net_value.item() + self.config['potential_weight'] * potential_value
		return np.tanh(combined_value)  # Tanh to keep it in [-1, 1]

	def _calculate_final_policy(self, root: _Node, temp: float) -> np.ndarray:
		visits = np.array([c.visit_count for c in root.children.values()], dtype=np.float32)
		actions = np.array(list(root.children.keys()))
		if temp == 0:
			policy = np.zeros_like(visits);
			policy[np.argmax(visits)] = 1.0
		else:
			policy = (visits ** (1 / temp)) / (np.sum(visits ** (1 / temp)) + 1e-8)
		full_policy = np.zeros(len(self.action_space), dtype=np.float32)
		full_policy[actions] = policy
		return full_policy


class ReplayBuffer:
	"""מאגר חזרות עם עדיפות (PER)."""

	def __init__(self, size: int, alpha: float = 0.6):
		self.buffer = []
		self.size = size
		self.alpha = alpha
		self.counter = itertools.count()

	def push(self, experience: Tuple, priority: Optional[float] = None):
		if priority is None:
			priority = max((abs(p) for p, _, _ in self.buffer), default=1.0)

		# Use negative priority because heapq is a min-heap
		heapq.heappush(self.buffer, (-priority, next(self.counter), experience))
		if len(self.buffer) > self.size:
			# Remove the element with the lowest priority (smallest absolute value)
			heapq.heappop(self.buffer)

	def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List, np.ndarray, np.ndarray]:
		if not self.buffer:
			return [], np.array([]), np.array([])

		priorities = np.array([-p for p, _, _ in self.buffer]) ** self.alpha
		probs = priorities / (priorities.sum() + 1e-8)

		# Ensure batch_size does not exceed buffer length
		actual_batch_size = min(batch_size, len(self.buffer))
		indices = np.random.choice(len(self.buffer), actual_batch_size, p=probs, replace=True)

		experiences = [self.buffer[i][2] for i in indices]

		weights = (len(self.buffer) * probs[indices]) ** (-beta)
		weights /= (weights.max() + 1e-8)

		return experiences, indices, weights

	def update_priorities(self, indices: np.ndarray, new_priorities: np.ndarray):
		for i, p in zip(indices, new_priorities):
			if i < len(self.buffer):
				_, count, exp = self.buffer[i]
				self.buffer[i] = (-p, count, exp)
		heapq.heapify(self.buffer)

	def __len__(self) -> int:
		return len(self.buffer)


# =================================================================
# 5. מאמן ראשי וקוריקולום
# =================================================================
class CurriculumManager:
	"""מנהל את רמת הקושי של האימון."""

	def __init__(self, config: Dict):
		self.start_difficulty = config['curriculum_start_difficulty']
		self.end_difficulty = Board.TOTAL_PEGS - 1
		self.win_rate_threshold = config['curriculum_win_rate_threshold']
		self.current_difficulty = self.start_difficulty
		self.win_rates = deque(maxlen=100)

	@staticmethod
	def _get_reverse_moves(board: Board) -> List:
		rev_moves = []
		for r_from, c_from in board.state:
			if board.get((r_from, c_from)) == 0:
				for dr, dc in Game.DIRECTIONS:
					r_to, c_to = r_from - dr, c_from - dc
					if board.get((r_to, c_to)) == 1 and board.get((r_from - dr // 2, c_from - dc // 2)) == 0:
						rev_moves.append(((r_from, c_from), (r_to, c_to)))
		return rev_moves

	@staticmethod
	def _apply_reverse_move(board: Board, move: Tuple):
		from_pos, to_pos = move
		over_pos = ((from_pos[0] + to_pos[0]) // 2, (from_pos[1] + to_pos[1]) // 2)
		board.set(from_pos, 1);
		board.set(over_pos, 1);
		board.set(to_pos, 0)

	def get_start_board(self) -> Board:
		goal_state = {pos: 0 for pos in Board.LEGAL_POSITIONS};
		goal_state[(3, 3)] = 1
		board = Board(state_dict=goal_state)
		for _ in range(self.current_difficulty):
			rev_moves = self._get_reverse_moves(board)
			if not rev_moves: break
			self._apply_reverse_move(board, random.choice(rev_moves))
		return board

	def update_difficulty(self, is_win: bool):
		self.win_rates.append(1 if is_win else 0)
		if len(self.win_rates) < self.win_rates.maxlen: return
		if np.mean(list(self.win_rates)) >= self.win_rate_threshold and self.current_difficulty < self.end_difficulty:
			self.current_difficulty += 1
			self.win_rates.clear()
			print(f"\n*** Curriculum difficulty increased to {self.current_difficulty} reverse moves! ***")


class UltimateTrainer:
	"""הלוגיקה המרכזית של האימון."""

	def __init__(self, model: PegSolitaireGNN, config: Dict):
		self.config = config
		self.device = config['device']
		self.model = model.to(self.device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'],
		                            weight_decay=config['weight_decay'])
		self.graph_encoder = GraphEncoder()
		self.potential_calculator = PotentialCalculator()
		self.env = PegSolitaireEnv(self.graph_encoder)
		self.mcts = MCTS(self.model, self.potential_calculator, config)
		self.replay_buffer = ReplayBuffer(config['buffer_size'], alpha=config.get('per_alpha', 0.6))
		self.curriculum_manager = CurriculumManager(config)

	def _self_play_game(self) -> Tuple[List, float]:
		start_board = self.curriculum_manager.get_start_board()
		self.env.reset(start_board=start_board)
		game_history = []
		while not self.env.done:
			temp = self.config['temperature'] if self.env.move_count < self.config['temp_threshold'] else 0.0
			policy = self.mcts.run(self.env, temperature=temp)
			if np.sum(policy) == 0: break
			game_history.append((self.env.get_observation(), policy))
			action_idx = np.random.choice(len(policy), p=policy)
			self.env.step(Game.ACTION_SPACE.from_index(action_idx))
		return game_history, self.env.get_game_score()

	def _update_network(self):
		if len(self.replay_buffer) < self.config['batch_size']: return 0.0, 0.0
		self.model.train()

		beta = self.config.get('per_beta', 0.4)  # Add beta annealing if needed
		experiences, indices, weights = self.replay_buffer.sample(self.config['batch_size'], beta)

		if not experiences: return 0.0, 0.0

		batch_data = [exp[0] for exp in experiences]
		target_policies = torch.tensor(np.array([exp[1] for exp in experiences]), dtype=torch.float32).to(self.device)
		target_values = torch.tensor([exp[2] for exp in experiences], dtype=torch.float32).to(self.device)
		importance_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

		data_loader = DataLoader(batch_data, batch_size=len(experiences))
		graph_batch = next(iter(data_loader)).to(self.device)

		pred_policies, pred_values = self.model(graph_batch)

		value_loss = (importance_weights * F.mse_loss(pred_values.squeeze(), target_values, reduction='none')).mean()
		policy_loss = F.kl_div(F.log_softmax(pred_policies, dim=1), target_policies, reduction='batchmean')
		total_loss = value_loss + policy_loss

		self.optimizer.zero_grad()
		total_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
		self.optimizer.step()

		# Update priorities
		td_errors = (pred_values.squeeze() - target_values).abs().detach().cpu().numpy()
		self.replay_buffer.update_priorities(indices, td_errors + 1e-6)

		return value_loss.item(), policy_loss.item()

	def train(self, num_iterations: int):
		for i in range(num_iterations):
			print(
				f"--- Iteration {i + 1}/{num_iterations} | Curriculum Difficulty: {self.curriculum_manager.current_difficulty} ---")
			self.model.eval()

			game_outcomes = []
			with tqdm(total=self.config['self_play_games_per_iter'], desc="Self-Playing") as pbar:
				for _ in range(self.config['self_play_games_per_iter']):
					history, score = self._self_play_game()
					game_outcomes.append({'history': history, 'score': score})
					pbar.update(1)

			if not game_outcomes: continue

			scores = [g['score'] for g in game_outcomes]
			if not scores: continue

			score_threshold = np.percentile(scores, self.config['ranked_reward_percentile'] * 100)
			for game in game_outcomes:
				is_win = game['score'] <= score_threshold
				z_val = 1.0 if is_win else -1.0
				self.curriculum_manager.update_difficulty(is_win)
				for obs, pi in game['history']:
					self.replay_buffer.push((obs, pi, z_val))

			with tqdm(total=self.config['training_steps_per_iter'], desc="Training") as pbar:
				for _ in range(self.config['training_steps_per_iter']):
					v_loss, p_loss = self._update_network()
					pbar.set_postfix({"v_loss": f"{v_loss:.3f}", "p_loss": f"{p_loss:.3f}"})
					pbar.update(1)

			if (i + 1) % self.config['save_interval'] == 0:
				torch.save(self.model.state_dict(), f"peg_solitaire_ultimate_model_iter_{i + 1}.pth")
				print(f"\nModel saved at iteration {i + 1}")


# =================================================================
# 6. GUI אינטראקטיבי (מגרסה 2)
# =================================================================
class PegSolitaireGUI(tk.Frame):
	CELL, R, PAD = 60, 22, 16
	PEG, HOLE, OUTL, HILITE = "#FFD600", "#202020", "#333", "#42A5F5"
	SUGGEST, BG = "#00C853", "#1e1e1e"
	TEXT_COLOR = "#e0e0e0"

	def __init__(self, master, agent: Optional[UltimateTrainer] = None):
		super().__init__(master, bg=self.BG)
		self.game = Game()
		self.agent = agent
		self.sel: Optional[Tuple[int, int]] = None
		self.hint: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None

		side = 7 * self.CELL + 2 * self.PAD
		self.canvas = tk.Canvas(self, width=side, height=side, bg=self.BG, highlightthickness=0)
		self.canvas.pack(pady=10)

		self.status = tk.Label(self, font=("Arial", 14), bg=self.BG, fg=self.TEXT_COLOR, anchor="w")
		self.status.pack(pady=5, fill="x", padx=10)

		btns = tk.Frame(self, bg=self.BG)
		btns.pack(pady=5)
		btn_style = {'bg': '#333', 'fg': self.TEXT_COLOR, 'activebackground': '#444', 'activeforeground': 'white',
		             'relief': 'flat', 'font': ('Arial', 10)}
		for txt, cmd in [("Undo", self.on_undo), ("Redo", self.on_redo), ("Reset", self.on_reset),
		                 ("Hint", self.on_hint)]:
			tk.Button(btns, text=txt, command=cmd, **btn_style).pack(side="left", padx=5)

		self.canvas.bind("<Button-1>", self.on_click)
		master.bind("<Control-z>", lambda e: self.on_undo())
		master.bind("<Control-y>", lambda e: self.on_redo())
		master.bind("<Control-r>", lambda e: self.on_reset())

		self.redraw()

	def _xy(self, pos: Tuple[int, int]) -> Tuple[int, int]:
		return (self.PAD + pos[1] * self.CELL + self.CELL // 2, self.PAD + pos[0] * self.CELL + self.CELL // 2)

	def redraw(self):
		self.canvas.delete("all")
		for pos in Board.LEGAL_POSITIONS:
			x, y = self._xy(pos)
			fill = self.PEG if self.game.board.get(pos) == 1 else self.HOLE
			outline = self.HILITE if pos == self.sel else self.OUTL
			self.canvas.create_oval(x - self.R, y - self.R, x + self.R, y + self.R, fill=fill, outline=outline, width=2)

		if self.hint:
			src, dst = self.hint
			x1, y1 = self._xy(src);
			x2, y2 = self._xy(dst)
			self.canvas.create_line(x1, y1, x2, y2, fill=self.SUGGEST, width=4, arrow=tk.LAST, arrowshape=(12, 15, 5))

		self._update_status()

	def on_click(self, event):
		pos = ((event.y - self.PAD) // self.CELL, (event.x - self.PAD) // self.CELL)
		if pos not in Board.LEGAL_POSITIONS: return

		if self.sel is None and self.game.board.get(pos) == 1:
			self.sel = pos
		elif self.sel and pos != self.sel:
			action = self._move_to_action(self.sel, pos)
			# Check if the generated action is in the list of legal moves
			if action and action in self.game.get_legal_moves():
				self.game.apply_action(action)
			self.sel = self.hint = None
		else:
			self.sel = None
		self.redraw()

	def _call_and_redraw(self, func):
		if func():
			self.sel = self.hint = None
			self.redraw()

	def on_undo(self):
		self._call_and_redraw(self.game.undo)

	def on_redo(self):
		self._call_and_redraw(self.game.redo)

	def on_reset(self):
		self.game.reset(); self._call_and_redraw(lambda: True)

	def on_hint(self):
		if not self.agent:
			messagebox.showinfo("Hint", "No agent loaded to provide a hint.")
			return
		if self.game.is_game_over(): return

		env = PegSolitaireEnv(self.agent.graph_encoder)
		env.game = self.game.clone()
		env.move_count = len(self.game.move_history)

		policy = self.agent.mcts.run(env, temperature=0.0)
		best_action_idx = np.argmax(policy)
		best_action = Game.ACTION_SPACE.from_index(best_action_idx)

		from_pos = (best_action[0], best_action[1])
		dr, dc = Game.DIRECTIONS[best_action[2]]
		to_pos = (from_pos[0] + dr, from_pos[1] + dc)

		self.hint = (from_pos, to_pos)
		self.redraw()

	def _move_to_action(self, src: Tuple, dst: Tuple) -> Optional[Tuple[int, int, int]]:
		dr, dc = dst[0] - src[0], dst[1] - src[1]
		if (dr, dc) in Game.DIRECTIONS:
			dir_idx = Game.DIRECTIONS.index((dr, dc))
			return (src[0], src[1], dir_idx)
		return None

	def _update_status(self):
		if self.game.is_win():
			text = "Victory! Solved."
		elif self.game.is_game_over():
			text = f"Game Over. Pegs left: {self.game.board.count_pegs()}"
		else:
			text = f"Pegs: {self.game.board.count_pegs()} | Moves: {len(self.game.move_history)}"
		self.status.config(text=text)


# =================================================================
# 7. הרצה ראשית
# =================================================================
if __name__ == '__main__':
	config = {
		'device': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
		'learning_rate': 0.0003, 'weight_decay': 1e-5, 'grad_clip': 1.0,
		'buffer_size': 50_000, 'batch_size': 256,
		'temperature': 1.0, 'temp_threshold': 10,
		'self_play_games_per_iter': 100,
		'training_steps_per_iter': 200,
		'save_interval': 10,
		'mcts_simulations': 100, 'c_puct': 2.5,
		'dirichlet_alpha': 0.3, 'noise_fraction': 0.25,
		'potential_weight': 0.15,  # משקל להנחיית החיפוש עם הפוטנציאל
		'ranked_reward_percentile': 0.5,
		'per_alpha': 0.6, 'per_beta': 0.4,
		'curriculum_start_difficulty': 2,
		'curriculum_win_rate_threshold': 0.7,
	}
	print(f"Using device: {config['device']}")

	model = PegSolitaireGNN()
	trainer = UltimateTrainer(model, config)

	MODEL_PATH = Path("peg_solitaire_ultimate_model.pth")
	if MODEL_PATH.exists():
		print(f"Loading existing model from {MODEL_PATH}...")
		model.load_state_dict(torch.load(MODEL_PATH, map_location=config['device']))
	else:
		print("No existing model found. Starting training from scratch.")
		trainer.train(num_iterations=100)
		torch.save(model.state_dict(), MODEL_PATH)
		print(f"Training finished. Model saved to {MODEL_PATH}")

	# Launch GUI
	root = tk.Tk()
	root.title("Ultimate Peg Solitaire AI")
	root.configure(bg=PegSolitaireGUI.BG)
	gui = PegSolitaireGUI(root, agent=trainer)
	gui.pack(padx=10, pady=10)
	root.mainloop()
