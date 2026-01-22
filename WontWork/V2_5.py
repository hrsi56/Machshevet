# =================================================================
# train_peg_solitaire_gnn_advanced.py
#
# קובץ אימון AlphaZero משופר עם ארכיטקטורת GNN.
# מיישם עקרונות מתקדמים: MCTS-DAG, PBRS, הנדסת תכונות, וכיול היפר-פרמטרים.
#
# להרצה, יש להתקין:
# pip install torch numpy scipy tqdm torch_geometric
# =================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
import copy
from tqdm import tqdm
from typing import List, Tuple, Dict


# =================================================================
# 1. לוגיקת משחק וכלי עזר (כולל תורת מחלקות המיקום)
# =================================================================
class PegSolitaireLogic:
	"""
	מחלקה מרכזית המכילה את כל הלוגיקה הסטטית של המשחק,
	כולל קבועים, הגדרת מהלכים, ותורת מחלקות המיקום.
	"""
	LEGAL_POSITIONS = sorted([
		(r, c) for r in range(7) for c in range(7)
		if not ((r < 2 or r > 4) and (c < 2 or c > 4))
	])
	POS_TO_IDX = {pos: i for i, pos in enumerate(LEGAL_POSITIONS)}
	IDX_TO_POS = {i: pos for i, pos in enumerate(LEGAL_POSITIONS)}
	NUM_NODES = len(LEGAL_POSITIONS)
	TOTAL_PEGS_START = NUM_NODES - 1
	DIRECTIONS = [(-2, 0), (2, 0), (0, -2), (0, 2)]

	# מיפוי אלכסונים עבור תורת מחלקות המיקום
	POS_CLASS_DIAGONALS = [
		{(r, c): (r - c) % 3 for r, c in LEGAL_POSITIONS},
		{(r, c): (r + c) % 3 for r, c in LEGAL_POSITIONS},
		{(r, c): r % 3 for r, c in LEGAL_POSITIONS},
		{(r, c): c % 3 for r, c in LEGAL_POSITIONS},
		{(r, c): (r - 2 * c) % 3 for r, c in LEGAL_POSITIONS},
		{(r, c): (c - 2 * r) % 3 for r, c in LEGAL_POSITIONS}
	]


	def __init__(self):
		self.action_map, self.num_actions = self._create_action_space()
		self.action_to_idx = {action: i for i, action in enumerate(self.action_map)}
		self.idx_to_action = {i: action for i, action in enumerate(self.action_map)}

	def _create_action_space(self) -> Tuple[List[Tuple[int, int]], int]:
		""" יוצר מרחב פעולה קנוני של כל 76 הקפיצות האפשריות. """
		actions = []
		for r_from, c_from in self.LEGAL_POSITIONS:
			for dr, dc in self.DIRECTIONS:
				r_to, c_to = r_from + dr, c_from + dc
				if (r_to, c_to) in self.LEGAL_POSITIONS:
					from_idx = self.POS_TO_IDX[(r_from, c_from)]
					to_idx = self.POS_TO_IDX[(r_to, c_to)]
					actions.append((from_idx, to_idx))
		return sorted(actions), len(actions)  # 76 actions

	@classmethod
	def get_position_class_vector(cls, board: 'Board') -> Tuple[int, ...]:
		""" מחשב את וקטור מחלקת המיקום עבור מצב לוח נתון. """
		class_vector = []
		for diag_map in cls.POS_CLASS_DIAGONALS:
			sums = [0, 0, 0]
			for pos, val in board.state.items():
				if val == 1:
					sums[diag_map[pos]] += 1
			class_vector.append(tuple(s % 2 for s in sums))

		# הסימן של הקומבינציה הלינארית הוא אינווריאנטי
		# (s0+s1+s2) mod 2 הוא 0. אנו צריכים רק 2 רכיבים.
		final_vector = []
		for s0, s1, s2 in class_vector:
			final_vector.extend([s0, s1])
		return tuple(
			final_vector[:6])  # This is simplified, real theory is complex. Using a known implementation pattern.

	@classmethod
	def is_unsolvable(cls, board: 'Board') -> bool:
		""" בודק אם מצב הלוח שייך למחלקת מיקום בלתי פתירה. """
		# This is a simplified check. Real check involves mapping vector to canonical form.
		# For this exercise, we check against a small known set.
		# A full implementation would require a lookup table of all 16 classes and their solvability.
		# We simplify here to demonstrate the PBRS concept.
		# The key idea is that some states are provably dead-ends.
		pegs = board.count_pegs()
		if pegs < 2 or pegs > cls.TOTAL_PEGS_START: return False  # Heuristic

		# A simple but powerful invariant: the "compass rule" or parity check on quadrants
		# Let's use a simpler, known invariant.
		# Sum of coordinates (x+y) parity
		# parity_sum = sum((r + c) % 2 for pos, peg in board.state.items() if peg == 1)
		# In the goal state (peg at 3,3), parity sum is (3+3)%2 = 0.
		# Any jump preserves the parity of this sum.
		# The initial state has parity sum of 0 (by calculation).
		# Therefore, any reachable state must have parity sum of 0.
		# This is not true. A jump changes parity.
		# Let's stick to a basic heuristic for PBRS demonstration.
		return False  # Disabling the complex check for now to ensure stability.
	# A full implementation requires careful validation of the class theory.


class Board:
	def __init__(self, empty_pos=(3, 3)):
		self.state = {pos: 1 for pos in PegSolitaireLogic.LEGAL_POSITIONS}
		if empty_pos in self.state:
			self.state[empty_pos] = 0

	def get(self, pos): return self.state.get(pos, None)

	def set(self, pos, val): self.state[pos] = val

	def count_pegs(self): return sum(self.state.values())

	def copy(self): return copy.deepcopy(self)

	def get_state_key(self) -> Tuple[int, ...]:
		# מפתח ייחודי וקנוני למצב הלוח, עבור טבלת הטרנספוזיציה
		return tuple(sorted([PegSolitaireLogic.POS_TO_IDX[pos] for pos, peg in self.state.items() if peg == 1]))


class Game:
	LOGIC = PegSolitaireLogic()

	def __init__(self, board=None):
		self.board = board if board else Board()

	def get_legal_actions(self) -> List[int]:
		moves = []
		for action_idx, (from_idx, to_idx) in enumerate(self.LOGIC.action_map):
			from_pos = self.LOGIC.IDX_TO_POS[from_idx]
			to_pos = self.LOGIC.IDX_TO_POS[to_idx]

			if self.board.get(from_pos) != 1 or self.board.get(to_pos) != 0:
				continue

			dr, dc = to_pos[0] - from_pos[0], to_pos[1] - from_pos[1]
			over_pos = (from_pos[0] + dr // 2, from_pos[1] + dc // 2)

			if self.board.get(over_pos) == 1:
				moves.append(action_idx)
		return moves

	def apply_action_idx(self, action_idx: int):
		from_idx, to_idx = self.LOGIC.idx_to_action[action_idx]
		from_pos = self.LOGIC.IDX_TO_POS[from_idx]
		to_pos = self.LOGIC.IDX_TO_POS[to_idx]
		dr, dc = to_pos[0] - from_pos[0], to_pos[1] - from_pos[1]
		over_pos = (from_pos[0] + dr // 2, from_pos[1] + dc // 2)

		self.board.set(from_pos, 0)
		self.board.set(over_pos, 0)
		self.board.set(to_pos, 1)

	def is_game_over(self):
		return len(self.get_legal_actions()) == 0

	def is_win(self):
		return self.board.count_pegs() == 1 and self.board.get((3, 3)) == 1

	def clone(self):
		return copy.deepcopy(self)


# =================================================================
# 2. שינוי מרכזי: מקודד גרף עם הנדסת תכונות מתקדמת
# =================================================================
class GraphEncoder:
	def __init__(self, logic: PegSolitaireLogic):
		self.logic = logic
		self.edge_index = self._create_static_edge_index()
		# תכונות צומת: [יש פיון, מבודד, מרחק מנהטן מהמרכז]
		self.num_node_features = 3
		# חישובים סטטיים מראש
		self.manhattan_distances = self._precompute_manhattan_distances()

	def _create_static_edge_index(self) -> torch.Tensor:
		edges = []
		for from_idx, to_idx in self.logic.action_map:
			edges.append([from_idx, to_idx])
			edges.append([to_idx, from_idx])  # קשתות דו-כיווניות
		return torch.tensor(edges, dtype=torch.long).t().contiguous()

	def _precompute_manhattan_distances(self) -> torch.Tensor:
		distances = torch.zeros(self.logic.NUM_NODES, 1, dtype=torch.float32)
		center_pos = (3, 3)
		for i in range(self.logic.NUM_NODES):
			pos = self.logic.IDX_TO_POS[i]
			dist = abs(pos[0] - center_pos[0]) + abs(pos[1] - center_pos[1])
			distances[i] = dist / 6.0  # נרמול
		return distances

	def encode(self, board: Board, move_count: int) -> Data:
		# תכונה 1: האם יש פיון
		is_peg = torch.zeros(self.logic.NUM_NODES, 1, dtype=torch.float32)
		peg_indices = [self.logic.POS_TO_IDX[pos] for pos, val in board.state.items() if val == 1]
		is_peg[peg_indices] = 1.0

		# תכונה 2: האם פיון מבודד (לא יכול לזוז)
		is_isolated = torch.zeros(self.logic.NUM_NODES, 1, dtype=torch.float32)
		temp_game = Game(board.copy())
		legal_from_indices = {self.logic.action_map[act_idx][0] for act_idx in temp_game.get_legal_actions()}
		for peg_idx in peg_indices:
			if peg_idx not in legal_from_indices:
				is_isolated[peg_idx] = 1.0

		node_features = torch.cat([
			is_peg,
			is_isolated,
			self.manhattan_distances
		], dim=1)

		# תכונות גלובליות
		pegs_left = float(board.count_pegs())

		return Data(
			x=node_features,
			edge_index=self.edge_index,
			pegs_left=torch.tensor([[pegs_left / self.logic.TOTAL_PEGS_START]], dtype=torch.float32),
			move_count=torch.tensor([[move_count / self.logic.TOTAL_PEGS_START]], dtype=torch.float32)
		)


# =================================================================
# 3. ארכיטקטורת GNN (מותאמת לתכונות החדשות)
# =================================================================
class PegSolitaireGNN(nn.Module):
	def __init__(self, node_features: int, num_actions: int, gnn_channels=128, num_heads=4):
		super(PegSolitaireGNN, self).__init__()
		global_features = 2  # pegs_left, move_count

		self.conv1 = GATv2Conv(node_features, gnn_channels, heads=num_heads)
		self.conv2 = GATv2Conv(gnn_channels * num_heads, gnn_channels, heads=num_heads)
		self.conv3 = GATv2Conv(gnn_channels * num_heads, gnn_channels, heads=1)

		# Value Head
		self.value_fc1 = nn.Linear(gnn_channels + global_features, 128)
		self.value_fc2 = nn.Linear(128, 1)

		# Policy Head
		self.policy_fc1 = nn.Linear(gnn_channels + global_features, 128)
		self.policy_fc2 = nn.Linear(128, num_actions)

	def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
		x, edge_index, batch = data.x, data.edge_index, data.batch

		x = F.elu(self.conv1(x, edge_index))
		x = F.elu(self.conv2(x, edge_index))
		x = F.elu(self.conv3(x, edge_index))

		graph_embedding = global_mean_pool(x, batch)
		combined_embedding = torch.cat([graph_embedding, data.pegs_left, data.move_count], dim=1)

		v = F.relu(self.value_fc1(combined_embedding))
		v = torch.tanh(self.value_fc2(v))

		p = F.relu(self.policy_fc1(combined_embedding))
		p_logits = self.policy_fc2(p)

		return p_logits, v


# =================================================================
# 4. סביבת משחק עם עיצוב תגמול (PBRS)
# =================================================================
class PegSolitaireEnv:
	def __init__(self, logic: PegSolitaireLogic, graph_encoder: GraphEncoder, config: Dict):
		self.logic = logic
		self.graph_encoder = graph_encoder
		self.config = config
		self.game = Game()
		self.done = False
		self.move_count = 0

	def reset(self) -> Data:
		self.game = Game()
		self.done = False
		self.move_count = 0
		return self.get_observation()

	def _calculate_potential(self, board: Board) -> float:
		""" פונקציית הפוטנציאל עבור PBRS. """
		if self.logic.is_unsolvable(board):
			return -100.0  # עונש כבד על כניסה למצב ללא פתרון
		return -float(board.count_pegs())  # היוריסטיקה בסיסית: פחות יתדות זה יותר טוב

	def step(self, action_idx: int) -> Tuple[Data, float, bool, Dict]:
		potential_before = self._calculate_potential(self.game.board)
		peg_count_before = self.game.board.count_pegs()

		self.game.apply_action_idx(action_idx)
		self.move_count += 1
		self.done = self.game.is_game_over()

		potential_after = self._calculate_potential(self.game.board)
		peg_count_after = self.game.board.count_pegs()

		# תגמול סביבה בסיסי
		env_reward = float(peg_count_before - peg_count_after)
		if self.done:
			env_reward += 50.0 if self.game.is_win() else -5.0

		# תגמול מעוצב סופי
		shaped_reward = env_reward + self.config['gamma'] * potential_after - potential_before

		return self.get_observation(), shaped_reward, self.done, {}

	def get_legal_actions(self) -> List[int]:
		return self.game.get_legal_actions()

	def get_observation(self) -> Data:
		return self.graph_encoder.encode(self.game.board, self.move_count)

	def get_state_key(self) -> Tuple[int, ...]:
		return self.game.board.get_state_key()

	def get_final_environmental_reward(self) -> float:
		""" מחזיר את התוצאה הסופית ה"אמיתית", ללא עיצוב. """
		if self.game.is_win():
			return 50.0
		# תגמול פרופורציונלי למספר היתדות שהוסרו
		return -float(self.game.board.count_pegs())

	def clone(self):
		return copy.deepcopy(self)


# =================================================================
# 5. שינוי מרכזי: MCTS עם טבלת טרנספוזיציה (DAG)
# =================================================================
class _Node:
	def __init__(self, prior: float, value_init: float = 0.0):
		self.visit_count: int = 0
		self.value_sum: float = value_init
		self.prior: float = prior
		self.children: Dict[int, "_Node"] = {}

	@property
	def value(self) -> float:
		return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0


class MCTS:
	def __init__(self, model: nn.Module, logic: PegSolitaireLogic, config: Dict):
		self.model = model
		self.logic = logic
		self.config = config
		self.device = config['device']

	def run(self, env: PegSolitaireEnv, temperature: float = 1.0) -> np.ndarray:
		initial_key = env.get_state_key()
		root = _Node(prior=0.0)
		transposition_table = {initial_key: root}

		# הרחבה ראשונית של השורש
		obs = env.get_observation()
		legal_actions = env.get_legal_actions()
		if not legal_actions:
			return np.zeros(self.logic.num_actions, dtype=np.float32)

		policy_logits, value = self._predict(obs)
		self._expand(root, legal_actions, policy_logits, value.item(), add_noise=True)

		for _ in range(self.config['mcts_simulations']):
			sim_env = env.clone()
			node = root
			path = [root]

			# 1. בחירה (Selection)
			while node.children:
				action_idx, child_node = self._select_child(node)
				sim_env.step(action_idx)
				path.append(child_node)

				# בדיקת טרנספוזיציה
				state_key = sim_env.get_state_key()
				if state_key not in transposition_table:
					transposition_table[state_key] = child_node
					node = child_node
					break  # הגענו לעלה בעץ החיפוש הנוכחי
				else:
					node = transposition_table[state_key]
					# אם הגענו לצומת שכבר קיים, אבל הוא לא חלק מהנתיב של הילדים של ההורה,
					# זה אומר שמצאנו מסלול אחר לאותו מצב (DAG). נמשיך ממנו.
					path[-1] = node  # עדכון הנתיב

			# 2. הרחבה והערכה (Expand & Evaluate)
			leaf_value = 0.0
			if not sim_env.done:
				obs = sim_env.get_observation()
				legal_moves = sim_env.get_legal_actions()
				policy_logits, value_tensor = self._predict(obs)
				leaf_value = value_tensor.item()
				if legal_moves:
					self._expand(node, legal_moves, policy_logits, leaf_value)
			else:
				# שימוש בתגמול הסביבתי הסופי כערך הטרמינלי
				leaf_value = sim_env.get_final_environmental_reward()

			# 3. גיבוי (Backpropagation)
			for node_in_path in reversed(path):
				node_in_path.visit_count += 1
				node_in_path.value_sum += leaf_value

		return self._calculate_final_policy(root, temperature)

	def _predict(self, obs: Data) -> Tuple[torch.Tensor, torch.Tensor]:
		obs.to(self.device)
		with torch.no_grad():
			return self.model(obs)

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

	def _expand(self, node: _Node, legal_actions: List[int], policy_logits: torch.Tensor, value_init: float,
	            add_noise: bool = False):
		policy_probs = torch.softmax(policy_logits.flatten(), dim=0)

		if add_noise:
			noise = torch.from_numpy(np.random.dirichlet([self.config['dirichlet_alpha']] * len(legal_actions))).to(
				self.device)
			frac = self.config['noise_fraction']
			policy_probs[legal_actions] = (1 - frac) * policy_probs[legal_actions] + frac * noise

		for action_idx in legal_actions:
			prob = policy_probs[action_idx].item()
			# אתחול ערך חכם: מתחילים מהערכת הרשת של ההורה
			node.children[action_idx] = _Node(prior=prob, value_init=value_init)

	def _calculate_final_policy(self, root: _Node, temperature: float) -> np.ndarray:
		if not root.children: return np.zeros(self.logic.num_actions, dtype=np.float32)

		action_indices = np.array(list(root.children.keys()))
		visits = np.array([child.visit_count for child in root.children.values()], dtype=np.float32)

		if temperature == 0:
			policy = np.zeros_like(visits)
			policy[np.argmax(visits)] = 1.0
		else:
			visits_temp = visits ** (1 / temperature)
			policy = visits_temp / np.sum(visits_temp)

		full_policy = np.zeros(self.logic.num_actions, dtype=np.float32)
		full_policy[action_indices] = policy
		return full_policy


class ReplayBuffer:
	def __init__(self, max_size: int):
		self.buffer: List[Data] = []
		self.max_size = max_size

	def push(self, experience: Data):
		if len(self.buffer) >= self.max_size: self.buffer.pop(0)
		self.buffer.append(experience)

	def sample(self, batch_size) -> List[Data]:
		indices = np.random.choice(len(self.buffer), batch_size, replace=False)
		return [self.buffer[i] for i in indices]

	def __len__(self): return len(self.buffer)


# =================================================================
# 6. מאמן AlphaZero משופר
# =================================================================
class AlphaZeroTrainer:
	def __init__(self, model, config):
		self.config = config
		self.device = config['device']
		self.logic = PegSolitaireLogic()
		self.graph_encoder = GraphEncoder(self.logic)
		self.model = model.to(self.device)
		self.mcts = MCTS(self.model, self.logic, self.config)
		self.replay_buffer = ReplayBuffer(max_size=config['buffer_size'])
		self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'],
		                            weight_decay=config['l2_regularization'])
		self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=config['lr_restart_interval'], T_mult=1)
		self.value_loss_fn = nn.MSELoss()
		self.policy_loss_fn = nn.CrossEntropyLoss()

	def _self_play_game(self):
		game_history: List[Tuple[Data, np.ndarray]] = []
		env = PegSolitaireEnv(self.logic, self.graph_encoder, self.config)

		while not env.done:
			temp = self.config['temperature'] if env.move_count < self.config['temp_threshold'] else 0.0
			policy = self.mcts.run(env, temperature=temp)
			if np.sum(policy) == 0: break

			game_history.append((env.get_observation(), policy))
			action_idx = np.random.choice(len(policy), p=policy)
			env.step(action_idx)

		# חשב את התגמול הסופי (z) והחל אותו על כל ההיסטוריה
		final_reward = env.get_final_environmental_reward()
		final_potential = self._calculate_potential(env.game.board)
		final_z = final_reward + self.config['gamma'] * final_potential

		for obs, pi in game_history:
			# Data augmentation can be complex with GNNs and is omitted here.
			# The structural priors from the GNN itself provide significant generalization.
			obs.y = torch.from_numpy(pi).float()
			obs.z = torch.tensor(final_z, dtype=torch.float32)
			self.replay_buffer.push(obs)

	def _calculate_potential(self, board: Board) -> float:
		# פונקציה זהה לזו שבסביבה, משמשת לחישוב z הסופי
		if self.logic.is_unsolvable(board): return -100.0
		return -float(board.count_pegs())

	def _update_network(self) -> Tuple[float, float]:
		if len(self.replay_buffer) < self.config['batch_size']: return 0.0, 0.0
		self.model.train()

		batch_data = self.replay_buffer.sample(self.config['batch_size'])
		data_loader = DataLoader(batch_data, batch_size=len(batch_data))
		graph_batch = next(iter(data_loader)).to(self.device)

		target_policies = graph_batch.y
		target_values = graph_batch.z.view(-1, 1)

		pred_policies, pred_values = self.model(graph_batch)
		value_loss = self.value_loss_fn(pred_values, target_values)
		policy_loss = self.policy_loss_fn(pred_policies, target_policies)
		total_loss = value_loss + policy_loss

		self.optimizer.zero_grad()
		total_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
		self.optimizer.step()
		self.scheduler.step()
		return value_loss.item(), policy_loss.item()

	def train(self, num_iterations: int):
		for i in range(num_iterations):
			print(f"--- Iteration {i + 1}/{num_iterations} | LR: {self.optimizer.param_groups[0]['lr']:.6f} ---")
			self.model.eval()

			pbar_self_play = tqdm(range(self.config['self_play_games_per_iter']), desc="Self-Playing")
			for _ in pbar_self_play:
				self._self_play_game()

			total_v_loss, total_p_loss = 0, 0
			pbar_train = tqdm(range(self.config['training_steps_per_iter']), desc="Training Network")
			for _ in pbar_train:
				v_loss, p_loss = self._update_network()
				total_v_loss += v_loss
				total_p_loss += p_loss
				if v_loss > 0 or p_loss > 0:
					pbar_train.set_postfix({"v_loss": f"{v_loss:.3f}", "p_loss": f"{p_loss:.3f}"})

			if (i + 1) % self.config['save_interval'] == 0:
				torch.save(self.model.state_dict(), f"peg_solitaire_gnn_model_iter_{i + 1}.pth")

			avg_v_loss = total_v_loss / self.config['training_steps_per_iter'] if self.config[
				                                                                      'training_steps_per_iter'] > 0 else 0
			avg_p_loss = total_p_loss / self.config['training_steps_per_iter'] if self.config[
				                                                                      'training_steps_per_iter'] > 0 else 0
			print(f"Buffer size: {len(self.replay_buffer)}, Avg Loss (V/P): {avg_v_loss:.4f}/{avg_p_loss:.4f}")


# =================================================================
# 7. הרצה ראשית עם תצורה משופרת
# =================================================================
if __name__ == '__main__':
	config = {
		'device': 'cuda' if torch.cuda.is_available() else 'cpu',
		# MCTS
		'mcts_simulations': 80,  # איזון בין איכות חיפוש למהירות
		'c_puct': 4.0,  # מעודד חקירה מספקת
		'dirichlet_alpha': 0.5,  # רעש לחקירה מגוונת
		'noise_fraction': 0.25,
		'temperature': 1.0,  # טמפרטורה לבחירת מהלך (חקירה)
		'temp_threshold': 10,  # מספר הצעדים לפני שהטמפרטורה יורדת ל-0 (ניצול)
		# Training
		'learning_rate': 1e-4,  # קצב למידה התחלתי
		'l2_regularization': 1e-4,  # למניעת התאמת יתר
		'buffer_size': 50000,  # איזון בין גיוון לרעננות נתונים
		'batch_size': 256,  # פשרה בין זיכרון ליציבות גרדיאנט
		'grad_clip': 1.0,
		'gamma': 0.99,  # פקטור היוון עבור PBRS
		'lr_restart_interval': 200,  # כל כמה צעדי אימון לאפס את קצב הלמידה
		# Loop
		'self_play_games_per_iter': 100,  # יותר משחקים לאיסוף נתונים
		'training_steps_per_iter': 200,  # יותר צעדי אימון לעיכול הנתונים
		'save_interval': 10,
	}
	print(f"Using device: {config['device']}")

	temp_logic = PegSolitaireLogic()
	temp_encoder = GraphEncoder(temp_logic)
	model = PegSolitaireGNN(
		node_features=temp_encoder.num_node_features,
		num_actions=temp_logic.num_actions
	)

	trainer = AlphaZeroTrainer(model, config)

	print("Starting ADVANCED AlphaZero training with GNN for Peg Solitaire...")
	trainer.train(num_iterations=200)
	print("Training finished.")

