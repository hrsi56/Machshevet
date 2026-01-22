# =================================================================
# train_peg_solitaire_final.py
#
# הקוד המלא והסופי. משלב GNN, תגמול מדורג (R2) ולמידת קוריקולום.
# להרצה, יש להתקין (ב-Colab, יש להריץ את זה בתא נפרד):
# !pip install torch_geometric
# !pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-$(python -c 'import torch; print(torch.__version__)').html
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
from typing import List, Tuple, Dict, Optional
from collections import deque


# =================================================================
# 1. מחלקות המשחק הבסיסיות (ללא שינוי)
# =================================================================
class Board:
	LEGAL_POSITIONS = sorted([
		(r, c) for r in range(7) for c in range(7)
		if not ((r < 2 or r > 4) and (c < 2 or c > 4))
	])
	TOTAL_PEGS = len(LEGAL_POSITIONS) - 1

	def __init__(self, state_dict=None, empty_pos=(3, 3)):
		if state_dict:
			self.state = state_dict.copy()
		else:
			self.state = {pos: 1 for pos in self.LEGAL_POSITIONS}
			if empty_pos in self.state:
				self.state[empty_pos] = 0

	def get(self, pos):
		return self.state.get(pos, None)

	def set(self, pos, value):
		self.state[pos] = value

	def count_pegs(self):
		return sum(self.state.values())

	def copy(self):
		return Board(state_dict=self.state)


class PegActionSpace:
	def __init__(self, board_size: int = 7):
		self.board_size = board_size
		self.directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
		self.num_actions = self.board_size * self.board_size * len(self.directions)
		self._action_to_index: Dict[Tuple[int, int, int], int] = {}
		self._index_to_action: List[Optional[Tuple[int, int, int]]] = [None] * self.num_actions
		idx = 0
		for r in range(board_size):
			for c in range(board_size):
				for dir_idx in range(len(self.directions)):
					action = (r, c, dir_idx)
					self._action_to_index[action] = idx
					self._index_to_action[idx] = action
					idx += 1

	def to_index(self, action):
		return self._action_to_index.get(action)

	def from_index(self, index):
		return self._index_to_action[index]

	def __len__(self):
		return self.num_actions


class Game:
	DIRECTIONS = [(-2, 0), (2, 0), (0, -2), (0, 2)]
	ACTION_SPACE = PegActionSpace()

	def __init__(self, board=None):
		self.board = board if board else Board()

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
		from_r, from_c, dir_idx = action
		dr, dc = self.DIRECTIONS[dir_idx]
		from_pos, to_pos = (from_r, from_c), (from_r + dr, from_c + dc)
		over_pos = (from_r + dr // 2, from_c + dc // 2)
		self.board.set(from_pos, 0);
		self.board.set(over_pos, 0);
		self.board.set(to_pos, 1)

	def is_game_over(self):
		return len(self.get_legal_moves()) == 0

	def clone(self):
		return copy.deepcopy(self)


# =================================================================
# 2. רכיבי GNN (ללא שינוי)
# =================================================================
class GraphEncoder:
	def __init__(self):
		self.node_map = {pos: i for i, pos in enumerate(Board.LEGAL_POSITIONS)}
		self.num_nodes = len(self.node_map)
		self.edge_index = self._create_static_edge_index()

	def _create_static_edge_index(self):
		edges = []
		for r, c in self.node_map:
			for dr, dc in Game.DIRECTIONS:
				if (r + dr, c + dc) in self.node_map:
					edges.append([self.node_map[(r, c)], self.node_map[(r + dr, c + dc)]])
		return torch.tensor(edges, dtype=torch.long).t().contiguous()

	def encode(self, board, move_count):
		node_features = np.zeros((self.num_nodes, 4), dtype=np.float32)
		for pos, i in self.node_map.items():
			node_features[i] = [board.get(pos) == 1, pos == (3, 3), pos[0] / 6., pos[1] / 6.]
		pegs = float(board.count_pegs())
		return Data(x=torch.from_numpy(node_features), edge_index=self.edge_index,
		            pegs_left=torch.tensor([[pegs / Board.TOTAL_PEGS]], dtype=torch.float32),
		            move_count=torch.tensor([[move_count / Board.TOTAL_PEGS]], dtype=torch.float32))


class PegSolitaireGNN(nn.Module):
	def __init__(self, node_features=4, num_actions=196, gnn_channels=64, num_heads=4):
		super().__init__()
		self.conv1 = GATv2Conv(node_features, gnn_channels, heads=num_heads, concat=True)
		self.conv2 = GATv2Conv(gnn_channels * num_heads, gnn_channels, heads=1, concat=False)
		self.value_fc1 = nn.Linear(gnn_channels + 2, 64)
		self.value_fc2 = nn.Linear(64, 1)
		self.policy_fc1 = nn.Linear(gnn_channels + 2, 64)
		self.policy_fc2 = nn.Linear(64, num_actions)

	def forward(self, data):
		x = F.elu(self.conv1(data.x, data.edge_index))
		x = F.elu(self.conv2(x, data.edge_index))
		graph_emb = global_mean_pool(x, data.batch)
		combined = torch.cat([graph_emb, data.pegs_left, data.move_count], dim=1)
		v = torch.tanh(self.value_fc2(F.relu(self.value_fc1(combined))))
		p = self.policy_fc2(F.relu(self.policy_fc1(combined)))
		return p, v


# =================================================================
# 3. רכיבי קוריקולום (ללא שינוי)
# =================================================================
class ReverseMoveGenerator:
	@staticmethod
	def get_reverse_moves(board: Board) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
		rev_moves = []
		for r_from, c_from in board.state:
			if board.get((r_from, c_from)) == 0:
				for dr, dc in Game.DIRECTIONS:
					r_over, c_over = r_from - dr // 2, c_from - dc // 2
					r_to, c_to = r_from - dr, c_from - dc
					if board.get((r_over, c_over)) == 0 and board.get((r_to, c_to)) == 1:
						rev_moves.append(((r_from, c_from), (r_to, c_to)))
		return rev_moves

	@staticmethod
	def apply_reverse_move(board: Board, move: Tuple[Tuple[int, int], Tuple[int, int]]):
		from_pos, to_pos = move
		over_pos = ((from_pos[0] + to_pos[0]) // 2, (from_pos[1] + to_pos[1]) // 2)
		board.set(from_pos, 1);
		board.set(over_pos, 1);
		board.set(to_pos, 0)


class CurriculumManager:
	def __init__(self, config):
		self.start_difficulty = config['curriculum_start_difficulty']
		self.end_difficulty = Board.TOTAL_PEGS
		self.win_rate_threshold = config['curriculum_win_rate_threshold']
		self.current_difficulty = self.start_difficulty
		self.win_rates = deque(maxlen=100)

	def get_start_board(self) -> Board:
		goal_state = {pos: 0 for pos in Board.LEGAL_POSITIONS}
		goal_state[(3, 3)] = 1
		board = Board(state_dict=goal_state)
		for _ in range(self.current_difficulty):
			rev_moves = ReverseMoveGenerator.get_reverse_moves(board)
			if not rev_moves: break
			move = random.choice(rev_moves)
			ReverseMoveGenerator.apply_reverse_move(board, move)
		return board

	def update_difficulty(self, is_win: bool) -> bool:
		self.win_rates.append(1 if is_win else 0)
		if len(self.win_rates) < self.win_rates.maxlen: return False
		current_win_rate = np.mean(list(self.win_rates))
		if current_win_rate >= self.win_rate_threshold and self.current_difficulty < self.end_difficulty:
			self.current_difficulty += 1
			self.win_rates.clear()
			print(f"\n*** Curriculum difficulty increased to {self.current_difficulty} reverse moves! ***")
			return True
		return False


# =================================================================
# 4. סביבה, MCTS, מאגר (ללא שינוי)
# =================================================================
class PegSolitaireEnv:
	def __init__(self, graph_encoder):
		self.graph_encoder = graph_encoder
		self.game: Game = Game()
		self.done: bool = False
		self.move_count: int = 0

	def reset(self, start_board: Optional[Board] = None):
		self.game = Game(board=start_board if start_board else Board())
		self.done = False;
		self.move_count = 0
		return self.get_observation()

	def step(self, action):
		self.game.apply_action(action);
		self.move_count += 1
		self.done = self.game.is_game_over()
		return self.get_observation(), 1.0, self.done, {}

	def get_legal_moves(self): return self.game.get_legal_moves()

	def get_observation(self): return self.graph_encoder.encode(self.game.board, self.move_count)

	def get_game_score(self): return float(self.game.board.count_pegs())

	def clone(self): return copy.deepcopy(self)


class _Node:
	def __init__(self, prior): self.visit_count = 0; self.value_sum = 0.0; self.prior = prior; self.children = {}

	@property
	def value(self): return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0


class MCTS:
	def __init__(self, game_env, model, action_space, config):
		self.game_env, self.model, self.action_space, self.config, self.device = \
			game_env, model, action_space, config, config['device']

	def run(self, temperature=1.0):
		root = _Node(0.0)
		self._expand(root, self.game_env.get_observation(), self.game_env.get_legal_moves(), add_noise=True)
		if not root.children: return np.zeros(len(self.action_space), dtype=np.float32)
		for _ in range(self.config['mcts_simulations']):
			game_copy = self.game_env.clone()
			node, path = root, [root]
			while node.children:
				action_idx, node = self._select_child(node)
				_, _, done, _ = game_copy.step(self.action_space.from_index(action_idx))
				path.append(node)
			value = self._evaluate_state(game_copy.get_observation()) if not done else 0.0
			if not done and game_copy.get_legal_moves():
				self._expand(node, game_copy.get_observation(), game_copy.get_legal_moves())
			for node_in_path in reversed(path):
				node_in_path.visit_count += 1;
				node_in_path.value_sum += value
		return self._calculate_final_policy(root, temperature)

	def _select_child(self, node):
		sqrt_total = np.sqrt(node.visit_count)
		best_s, best_a, best_c = -np.inf, -1, None
		for a_idx, child in node.children.items():
			score = child.value + self.config['c_puct'] * child.prior * sqrt_total / (1 + child.visit_count)
			if score > best_s: best_s, best_a, best_c = score, a_idx, child
		return best_a, best_c

	def _expand(self, node, obs, legal_actions, add_noise=False):
		obs.to(self.device)
		with torch.no_grad():
			logits, _ = self.model(obs)
			probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
		if add_noise and legal_actions:
			noise = np.random.dirichlet([self.config['dirichlet_alpha']] * len(legal_actions))
			for i, action in enumerate(legal_actions):
				idx = self.action_space.to_index(action)
				probs[idx] = probs[idx] * (1 - self.config['noise_fraction']) + self.config['noise_fraction'] * noise[i]
		for action in legal_actions:
			node.children[self.action_space.to_index(action)] = _Node(probs[self.action_space.to_index(action)])

	def _evaluate_state(self, obs):
		obs.to(self.device)
		with torch.no_grad(): _, val = self.model(obs); return val.item()

	def _calculate_final_policy(self, root, temp):
		visits = np.array([c.visit_count for c in root.children.values()], dtype=np.float32)
		actions = np.array(list(root.children.keys()))
		if temp == 0:
			policy = np.zeros_like(visits); policy[np.argmax(visits)] = 1.0
		else:
			policy = (visits ** (1 / temp)) / np.sum(visits ** (1 / temp))
		full_policy = np.zeros(len(self.action_space), dtype=np.float32)
		full_policy[actions] = policy
		return full_policy


class ReplayBuffer:
	def __init__(self, size): self.buffer, self.size = [], size

	def push(self, exp):
		if len(self.buffer) >= self.size: self.buffer.pop(0)
		self.buffer.append(exp)

	def sample(self, batch_size): return random.sample(self.buffer, batch_size)

	def __len__(self): return len(self.buffer)


# =================================================================
# 5. מאמן AlphaZero עם התיקון
# =================================================================
class AlphaZeroTrainer:
	def __init__(self, model, config):
		self.config = config;
		self.device = config['device']
		self.model = model.to(self.device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'],
		                            weight_decay=config['weight_decay'])
		self.graph_encoder = GraphEncoder()
		self.env = PegSolitaireEnv(self.graph_encoder)
		self.mcts = MCTS(self.env, self.model, Game.ACTION_SPACE, config)
		self.replay_buffer = ReplayBuffer(config['buffer_size'])
		self.game_outcomes = deque(maxlen=config['ranked_reward_buffer_size'])
		self.curriculum_manager = CurriculumManager(config)

	def _self_play_game(self):
		start_board = self.curriculum_manager.get_start_board()
		self.env.reset(start_board=start_board)
		game_history = []
		while not self.env.done:
			temp = self.config['temperature'] if self.env.move_count < self.config['temp_threshold'] else 0.0
			policy = self.mcts.run(temperature=temp)
			if np.sum(policy) == 0: break
			game_history.append((self.env.get_observation(), policy))
			action_idx = np.random.choice(len(policy), p=policy)
			self.env.step(Game.ACTION_SPACE.from_index(action_idx))
		return game_history, self.env.get_game_score()

	def _update_network(self):
		if len(self.replay_buffer) < self.config['batch_size']: return 0.0, 0.0
		self.model.train()
		batch = self.replay_buffer.sample(self.config['batch_size'])
		data_loader = DataLoader(batch, batch_size=self.config['batch_size'], shuffle=True)
		graph_batch = next(iter(data_loader)).to(self.device)

		pred_policies, pred_values = self.model(graph_batch)
		value_loss = F.mse_loss(pred_values.squeeze(), graph_batch.z)
		policy_loss = F.cross_entropy(pred_policies, graph_batch.y)
		total_loss = value_loss + policy_loss

		self.optimizer.zero_grad();
		total_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
		self.optimizer.step()
		return value_loss.item(), policy_loss.item()

	def train(self, num_iterations: int):
		for i in range(num_iterations):
			print(
				f"--- Iteration {i + 1}/{num_iterations} | Curriculum Difficulty: {self.curriculum_manager.current_difficulty} ---")
			self.model.eval()

			iter_games = []
			wins_this_iter = 0
			with tqdm(total=self.config['self_play_games_per_iter'], desc="Self-Playing") as pbar:
				for game_idx in range(self.config['self_play_games_per_iter']):
					history, score = self._self_play_game()
					iter_games.append({'history': history, 'score': score})
					self.game_outcomes.append(score)

					# מעקב אחרי מספר הניצחונות (ציון = 1) באיטרציה הנוכחית
					if score == 1.0:
						wins_this_iter += 1

					# חישוב ועדכון אחוז הניצחון בסרגל ההתקדמות
					win_perc = (wins_this_iter / (game_idx + 1)) * 100
					pbar.set_postfix({"solve_perc": f"{win_perc:.1f}%"})
					pbar.update(1)

			if len(self.game_outcomes) > 0:
				score_threshold = np.percentile(list(self.game_outcomes), self.config['ranked_reward_percentile'] * 100)

				for game in iter_games:
					is_win = game['score'] <= score_threshold
					z_val = 1.0 if is_win else -1.0
					self.curriculum_manager.update_difficulty(is_win)
					for obs, pi in game['history']:
						# ================== התיקון נמצא כאן ==================
						# שינוי הצורה של וקטור המדיניות לדו-ממדית [1, num_actions]
						# כדי שה-DataLoader יערום את הווקטורים נכון.
						obs.y = torch.tensor(pi, dtype=torch.float32).unsqueeze(0)
						# =====================================================
						obs.z = torch.tensor(z_val, dtype=torch.float32)
						self.replay_buffer.push(obs)

			with tqdm(total=self.config['training_steps_per_iter'], desc="Training") as pbar:
				for _ in range(self.config['training_steps_per_iter']):
					v_loss, p_loss = self._update_network()
					pbar.set_postfix({"v_loss": f"{v_loss:.3f}", "p_loss": f"{p_loss:.3f}"})
					pbar.update(1)

			if (i + 1) % self.config['save_interval'] == 0:
				torch.save(self.model.state_dict(), f"peg_solitaire_final_model_iter_{i + 1}.pth")


# =================================================================
# 6. הרצה ראשית (ללא שינוי)
# =================================================================
if __name__ == '__main__':
	config = {
		'device': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
		'learning_rate': 0.001, 'weight_decay': 1e-5, 'grad_clip': 1.0,
		'buffer_size': 50_000, 'batch_size': 256,
		'temperature': 1.0, 'temp_threshold': 10,
		'self_play_games_per_iter': 100,
		'training_steps_per_iter': 200,
		'save_interval': 5,
		'mcts_simulations': 80, 'c_puct': 2.5,
		'dirichlet_alpha': 0.3, 'noise_fraction': 0.25,
		'ranked_reward_percentile': 0.5,
		'ranked_reward_buffer_size': 2000,
		'curriculum_start_difficulty': 1,
		'curriculum_win_rate_threshold': 0.7,
	}
	print(f"Using device: {config['device']}")

	model = PegSolitaireGNN(num_actions=Game.ACTION_SPACE.num_actions)
	trainer = AlphaZeroTrainer(model, config)

	print("Starting Final AlphaZero training (GNN + R2 + Curriculum)...")
	trainer.train(num_iterations=500)
	print("Training finished.")
