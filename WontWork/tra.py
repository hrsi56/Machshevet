# =================================================================
# train_peg_solitaire.py
#
# קובץ אימון מלא למערכת AlphaZero לפתרון פג סוליטר
# להרצה, יש להתקין:
# pip install torch numpy scipy tqdm
# =================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import math
from scipy.ndimage import convolve
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional, Callable, Union
import heapq
import itertools


# =================================================================
# 1. מחלקת הלוח (Board)
# =================================================================
class Board:
	LEGAL_POSITIONS = sorted([
		(r, c) for r in range(7) for c in range(7)
		if not ((r < 2 or r > 4) and (c < 2 or c > 4))
	])
	LEGAL_MASK = np.zeros((7, 7), dtype=bool)
	for r, c in LEGAL_POSITIONS:
		LEGAL_MASK[r, c] = True

	def __init__(self, empty_pos=(3, 3)):
		self.state = {pos: 1 for pos in self.LEGAL_POSITIONS}
		if empty_pos in self.state:
			self.state[empty_pos] = 0

	def get(self, pos):
		return self.state.get(pos, None)

	def set(self, pos, value):
		if pos in self.LEGAL_POSITIONS:
			self.state[pos] = value

	def all_pegs(self):
		return [pos for pos, val in self.state.items() if val == 1]

	def count_pegs(self):
		return sum(self.state.values())

	def as_array(self):
		arr = np.full((7, 7), -1.0, dtype=np.float32)
		for pos, val in self.state.items():
			arr[pos] = val
		return arr

	def copy(self):
		new_board = Board()
		new_board.state = self.state.copy()
		return new_board

	def __hash__(self):
		return hash(tuple(sorted(self.state.items())))

	def __eq__(self, other):
		return isinstance(other, Board) and self.state == other.state


# =================================================================
# 2. מחלקת מרחב הפעולות (ActionSpace)
# =================================================================
class PegActionSpace:
	def __init__(self, board_size: int = 7):
		self.board_size = board_size
		self.directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
		self.num_directions = len(self.directions)
		self.num_actions = self.board_size * self.board_size * self.num_directions
		self._action_to_index: Dict[Tuple[int, int, int], int] = {}
		self._index_to_action: List[Tuple[int, int, int]] = [None] * self.num_actions
		idx = 0
		for r in range(self.board_size):
			for c in range(self.board_size):
				for dir_idx in range(self.num_directions):
					action = (r, c, dir_idx)
					self._action_to_index[action] = idx
					self._index_to_action[idx] = action
					idx += 1

	def to_index(self, action: Tuple[int, int, int]) -> int:
		return self._action_to_index.get(action)

	def from_index(self, index: int) -> Tuple[int, int, int]:
		return self._index_to_action[index]

	def action_to_move(self, action: Tuple[int, int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
		from_r, from_c, dir_idx = action
		dr, dc = self.directions[dir_idx]
		to_pos = (from_r + dr, from_c + dc)
		from_pos = (from_r, from_c)
		return from_pos, to_pos

	def __len__(self) -> int:
		return self.num_actions

	def __getitem__(self, index: int) -> Tuple[int, int, int]:
		return self.from_index(index)


# =================================================================
# 3. מחלקת לוגיקת המשחק (Game)
# =================================================================
class Game:
	DIRECTIONS = [(-2, 0), (2, 0), (0, -2), (0, 2)]

	def __init__(self, board=None):
		self.board = board if board else Board()

	def is_legal_move(self, from_pos, to_pos):
		if self.board.get(from_pos) != 1 or self.board.get(to_pos) != 0:
			return False, None
		dr, dc = to_pos[0] - from_pos[0], to_pos[1] - from_pos[1]
		if (abs(dr), abs(dc)) not in [(2, 0), (0, 2)]:
			return False, None
		over_pos = (from_pos[0] + dr // 2, from_pos[1] + dc // 2)
		if self.board.get(over_pos) != 1:
			return False, None
		return True, over_pos

	def get_legal_moves(self) -> List[Tuple[int, int, int]]:
		moves = []
		for r_from in range(7):
			for c_from in range(7):
				from_pos = (r_from, c_from)
				if self.board.get(from_pos) == 1:
					for dir_idx, d in enumerate(self.DIRECTIONS):
						to_pos = (from_pos[0] + d[0], from_pos[1] + d[1])
						is_legal, _ = self.is_legal_move(from_pos, to_pos)
						if is_legal:
							moves.append((r_from, c_from, dir_idx))
		return moves

	def apply_action(self, action: Tuple[int, int, int]):
		from_pos, to_pos = PegActionSpace().action_to_move(action)
		is_legal, over_pos = self.is_legal_move(from_pos, to_pos)
		if not is_legal:
			raise ValueError("Illegal move passed to apply_action")
		self.board.set(from_pos, 0)
		self.board.set(over_pos, 0)
		self.board.set(to_pos, 1)

	def is_game_over(self):
		return len(self.get_legal_moves()) == 0

	def is_win(self):
		return self.board.count_pegs() == 1 and self.board.get((3, 3)) == 1

	def reset(self, board=None):
		self.board = board if board else Board()

	def clone(self):
		return copy.deepcopy(self)


# =================================================================
# 4. סביבת המשחק (PegSolitaireEnv)
# =================================================================
class PegSolitaireEnv:
	def __init__(self):
		self.game = Game()
		self.done = False
		self.TOTAL_PEGS = 32

	def reset(self, state_dict=None):
		board = Board()
		if state_dict:
			board.state = state_dict
		self.game.reset(board)
		self.done = False
		return self.encode_observation()

	def step(self, action: Tuple[int, int, int]):
		self.game.apply_action(action)
		self.done = self.game.is_game_over()
		# The reward logic is simplified here; a full system might use potential-based shaping
		reward = 0.0
		if self.done:
			reward = 1.0 if self.game.is_win() else -1.0
		return self.encode_observation(), reward, self.done, {}

	def get_legal_moves(self):
		return self.game.get_legal_moves()

	def is_win(self):
		return self.game.is_win()

	def encode_observation(self) -> np.ndarray:
		board_array = self.game.board.as_array()
		pegs = (board_array == 1).astype(np.float32)
		holes = (board_array == 0).astype(np.float32)
		valid = (board_array != -1).astype(np.float32)
		return np.stack([pegs, holes, valid], axis=-1)

	def get_final_reward(self, is_win: bool) -> float:
		return 1000.0 if is_win else -200.0

	def clone(self):
		return copy.deepcopy(self)


# =================================================================
# 5. הרשת העצבית (PegSolitaireNetV2)
# =================================================================
class ResidualBlock(nn.Module):
	def __init__(self, num_channels: int):
		super(ResidualBlock, self).__init__()
		self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm2d(num_channels)
		self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
		self.bn2 = nn.BatchNorm2d(num_channels)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		residual = x
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += residual
		out = F.relu(out)
		return out


class PegSolitaireNetV2(nn.Module):
	def __init__(self, input_shape=(3, 7, 7), num_actions=196, num_res_blocks=4, num_channels=64):
		super(PegSolitaireNetV2, self).__init__()
		in_channels, height, width = input_shape
		self.conv_in = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1)
		self.bn_in = nn.BatchNorm2d(num_channels)
		self.res_blocks = nn.ModuleList([ResidualBlock(num_channels) for _ in range(num_res_blocks)])
		self.flattened_size = num_channels * height * width
		self.fc_value1 = nn.Linear(self.flattened_size, 64)
		self.fc_value2 = nn.Linear(64, 1)
		self.fc_policy1 = nn.Linear(self.flattened_size, 64)
		self.fc_policy2 = nn.Linear(64, num_actions)

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		x = x.permute(0, 3, 1, 2).contiguous()  # (B, H, W, C) -> (B, C, H, W)
		x = F.relu(self.bn_in(self.conv_in(x)))
		for block in self.res_blocks:
			x = block(x)
		x_flat = x.reshape(-1, self.flattened_size)
		v = F.relu(self.fc_value1(x_flat))
		v = torch.tanh(self.fc_value2(v))
		p = F.relu(self.fc_policy1(x_flat))
		p_logits = self.fc_policy2(p)
		return p_logits, v


# =================================================================
# 6. מחלקות MCTS ומאגר הזיכרון
# =================================================================
class _Node:
	def __init__(self, prior: float):
		self.visit_count: int = 0
		self.value_sum: float = 0.0
		self.prior: float = prior
		self.children: Dict[int, "_Node"] = {}

	@property
	def value(self) -> float:
		if self.visit_count == 0: return 0.0
		return self.value_sum / self.visit_count


class MCTS:
	def __init__(self, game_env, model, action_space, config):
		self.game_env = game_env
		self.model = model
		self.action_space = action_space
		self.config = config
		self.device = config['device']

	def run(self, temperature: float = 1.0) -> np.ndarray:
		root = _Node(prior=0.0)
		current_obs = self.game_env.encode_observation()
		legal_actions = self.game_env.get_legal_moves()
		self._expand(root, current_obs, legal_actions, add_noise=True)
		for _ in range(self.config['mcts_simulations']):
			game_copy = self.game_env.clone()
			node = root
			path = [root]
			done = False
			while node.children:
				action_idx, node = self._select_child(node)
				path.append(node)
				_, _, done, _ = game_copy.step(self.action_space[action_idx])
			value = 0.0
			if not done:
				obs = game_copy.encode_observation()
				self._expand(node, obs, game_copy.get_legal_moves())
				value = self._evaluate_state(obs)
			else:
				value = game_copy.get_final_reward(is_win=game_copy.is_win())
			for node_in_path in reversed(path):
				node_in_path.visit_count += 1
				node_in_path.value_sum += value
		return self._calculate_final_policy(root, temperature)

	def _select_child(self, node: _Node) -> Tuple[int, _Node]:
		sqrt_total_visits = np.sqrt(node.visit_count)
		best_score, best_action_idx, best_child = -np.inf, -1, None
		for action_idx, child_node in node.children.items():
			q_value = child_node.value
			u_value = self.config['c_puct'] * child_node.prior * sqrt_total_visits / (1 + child_node.visit_count)
			score = q_value + u_value
			if score > best_score:
				best_score, best_action_idx, best_child = score, action_idx, child_node
		return best_action_idx, best_child

	def _expand(self, node: _Node, obs: np.ndarray, legal_actions: List[Tuple], add_noise: bool = False):
		with torch.no_grad():
			obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
			policy_logits, _ = self.model(obs_tensor)
			policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy().flatten()
		if add_noise:
			noise = np.random.dirichlet([self.config['dirichlet_alpha']] * len(legal_actions))
			for i, action in enumerate(legal_actions):
				action_idx = self.action_space.to_index(action)
				policy_probs[action_idx] = (1 - self.config['noise_fraction']) * policy_probs[action_idx] + self.config[
					'noise_fraction'] * noise[i]
		for action in legal_actions:
			action_idx = self.action_space.to_index(action)
			node.children[action_idx] = _Node(prior=policy_probs[action_idx])

	def _evaluate_state(self, obs: np.ndarray) -> float:
		with torch.no_grad():
			obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
			_, value_tensor = self.model(obs_tensor)
			return value_tensor.item()

	def _calculate_final_policy(self, root: _Node, temperature: float) -> np.ndarray:
		visits = np.array([child.visit_count for child in root.children.values()], dtype=np.float32)
		action_indices = np.array(list(root.children.keys()))
		if temperature == 0:
			policy = np.zeros_like(visits)
			policy[np.argmax(visits)] = 1.0
		else:
			visits_temp = visits ** (1 / temperature)
			policy = visits_temp / np.sum(visits_temp)
		full_policy = np.zeros(len(self.action_space), dtype=np.float32)
		full_policy[action_indices] = policy
		return full_policy


class ReplayBuffer:
	def __init__(self, max_size: int):
		self.buffer = []
		self.max_size = max_size

	def push(self, experience):
		if len(self.buffer) >= self.max_size:
			self.buffer.pop(0)
		self.buffer.append(experience)

	def sample(self, batch_size):
		indices = np.random.choice(len(self.buffer), batch_size, replace=False)
		return [self.buffer[i] for i in indices]

	def __len__(self):
		return len(self.buffer)


# =================================================================
# 7. מחלקת האימון (AlphaZeroTrainer)
# =================================================================
class AlphaZeroTrainer:
	def __init__(self, model, config):
		self.config = config
		self.device = config['device']
		self.model = model.to(self.device)
		self.action_space = PegActionSpace()
		self.env = PegSolitaireEnv()
		self.mcts = MCTS(self.env, self.model, self.action_space, self.config)
		self.replay_buffer = ReplayBuffer(max_size=config['buffer_size'])
		self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'],
		                            weight_decay=config['weight_decay'])
		self.value_loss_fn = nn.MSELoss()
		self.policy_loss_fn = nn.CrossEntropyLoss()

	def _self_play_game(self):
		game_history = []
		self.env.reset()
		while not self.env.done:
			policy = self.mcts.run(temperature=self.config['temperature'])
			current_obs = self.env.encode_observation()
			game_history.append([current_obs, policy, 0.0])
			action_idx = np.random.choice(len(policy), p=policy)
			self.env.step(self.action_space.from_index(action_idx))
		final_reward = self.env.get_final_reward(self.env.is_win())
		for experience in game_history:
			experience[2] = final_reward
		return game_history

	def _update_network(self):
		if len(self.replay_buffer) < self.config['batch_size']: return 0.0, 0.0
		self.model.train()
		batch = self.replay_buffer.sample(self.config['batch_size'])
		states = torch.as_tensor(np.array([exp[0] for exp in batch]), dtype=torch.float32, device=self.device)
		target_policies = torch.as_tensor(np.array([exp[1] for exp in batch]), dtype=torch.float32, device=self.device)
		target_values = torch.as_tensor(np.array([exp[2] for exp in batch]), dtype=torch.float32, device=self.device)

		pred_policies, pred_values = self.model(states)
		value_loss = self.value_loss_fn(pred_values.squeeze(), target_values)
		policy_loss = self.policy_loss_fn(pred_policies, target_policies)
		total_loss = value_loss + policy_loss

		self.optimizer.zero_grad()
		total_loss.backward()
		self.optimizer.step()
		return value_loss.item(), policy_loss.item()

	def train(self, num_iterations: int):
		for i in range(num_iterations):
			print(f"--- Iteration {i + 1}/{num_iterations} ---")
			self.model.eval()
			with tqdm(total=self.config['self_play_games_per_iter'], desc="Self-Playing") as pbar:
				for _ in range(self.config['self_play_games_per_iter']):
					game_data = self._self_play_game()
					for exp in game_data:
						self.replay_buffer.push(exp)
					pbar.update(1)

			with tqdm(total=self.config['training_steps_per_iter'], desc="Training Network") as pbar:
				for _ in range(self.config['training_steps_per_iter']):
					self._update_network()
					pbar.update(1)

			if (i + 1) % self.config['save_interval'] == 0:
				torch.save(self.model.state_dict(), f"peg_solitaire_model_iter_{i + 1}.pth")
			print(f"Buffer size: {len(self.replay_buffer)}")


# =================================================================
# 8. הרצה ראשית
# =================================================================
if __name__ == '__main__':
	# הגדרת היפר-פרמטרים
	config = {
		'device': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
		'learning_rate': 0.0005,
		'weight_decay': 1e-4,
		'buffer_size': 50_000,
		'batch_size': 256,
		'temperature': 1.0,  # טמפרטורה למשחק עצמי (גבוה=חקירה, נמוך=ניצול)
		'self_play_games_per_iter': 100,  # מספר משחקים לייצר בכל איטרציה
		'training_steps_per_iter': 200,  # מספר צעדי אימון על הדאטה החדש
		'save_interval': 5,  # כל כמה איטרציות לשמור את המודל
		'mcts_simulations': 80,  # מספר סימולציות MCTS לכל מהלך
		'c_puct': 2.0,  # פרמטר חקירה ב-MCTS
		'dirichlet_alpha': 0.3,
		'noise_fraction': 0.25,
	}
	print(f"Using device: {config['device']}")

	# יצירת המודל והמאמן
	model = PegSolitaireNetV2()
	trainer = AlphaZeroTrainer(model, config)

	# התחלת האימון
	print("Starting AlphaZero training for Peg Solitaire...")
	trainer.train(num_iterations=200)
	print("Training finished.")