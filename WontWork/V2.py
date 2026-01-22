# =================================================================
# train_peg_solitaire_gnn.py
#
# קובץ אימון AlphaZero עם ארכיטקטורת Graph Neural Network (GNN)
# להרצה, יש להתקין:
# pip install torch numpy scipy tqdm torch_geometric
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
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional

# =================================================================
# 1. מחלקות המשחק הבסיסיות (ללא שינוי מהותי)
# =================================================================
class Board:
    LEGAL_POSITIONS = sorted([
       (r, c) for r in range(7) for c in range(7)
       if not ((r < 2 or r > 4) and (c < 2 or c > 4))
    ])
    TOTAL_PEGS = len(LEGAL_POSITIONS) - 1

    def __init__(self, empty_pos=(3, 3)):
       self.state = {pos: 1 for pos in self.LEGAL_POSITIONS}
       if empty_pos in self.state:
          self.state[empty_pos] = 0

    def get(self, pos):
       return self.state.get(pos, None)

    def set(self, pos, value):
       if pos in self.LEGAL_POSITIONS:
          self.state[pos] = value

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

class PegActionSpace:
    def __init__(self, board_size: int = 7):
        self.board_size = board_size
        self.directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
        self.num_directions = len(self.directions)
        self.num_actions = self.board_size * self.board_size * self.num_directions
        self._action_to_index: Dict[Tuple[int, int, int], int] = {}
        self._index_to_action: List[Optional[Tuple[int, int, int]]] = [None] * self.num_actions
        idx = 0
        for r in range(self.board_size):
            for c in range(self.board_size):
                for dir_idx in range(self.num_directions):
                    action = (r, c, dir_idx)
                    self._action_to_index[action] = idx
                    self._index_to_action[idx] = action
                    idx += 1

    def to_index(self, action: Tuple[int, int, int]) -> Optional[int]:
       return self._action_to_index.get(action)

    def from_index(self, index: int) -> Tuple[int, int, int]:
       return self._index_to_action[index]

    def action_to_move(self, action: Tuple[int, int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
       from_r, from_c, dir_idx = action
       dr, dc = self.directions[dir_idx]
       return (from_r, from_c), (from_r + dr, from_c + dc)

    def __len__(self) -> int:
       return self.num_actions

class Game:
    DIRECTIONS = [(-2, 0), (2, 0), (0, -2), (0, 2)]
    ACTION_SPACE = PegActionSpace()

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
       from_pos, to_pos = self.ACTION_SPACE.action_to_move(action)
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

    def clone(self):
       return copy.deepcopy(self)

# =================================================================
# 2. שינוי מרכזי: ייצוג גרפי של הלוח
# =================================================================
class GraphEncoder:
    """
    מחלקה שמטרתה להמיר מצב לוח לייצוג גרפי המתאים ל-torch_geometric.
    הגרף עצמו (קשתות) סטטי, רק תכונות הצמתים משתנות עם מצב הלוח.
    """
    def __init__(self):
        self.node_map = {pos: i for i, pos in enumerate(Board.LEGAL_POSITIONS)}
        self.num_nodes = len(self.node_map)
        self.edge_index = self._create_static_edge_index()
        # תכונות צומת: [האם יש פיון?, האם זה המרכז?, קואורדינטה R, קואורדינטה C]
        self.num_node_features = 4

    def _create_static_edge_index(self) -> torch.Tensor:
        """
        יוצר את רשימת הקשתות הסטטית של הלוח.
        קשת קיימת בין שני צמתים אם הם במרחק קפיצה זה מזה.
        """
        edges = []
        for r_from, c_from in self.node_map.keys():
            for dr, dc in Game.DIRECTIONS:
                r_to, c_to = r_from + dr, c_from + dc
                if (r_to, c_to) in self.node_map:
                    u = self.node_map[(r_from, c_from)]
                    v = self.node_map[(r_to, c_to)]
                    edges.append([u, v])
        # [2, num_edges]
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def encode(self, board: Board, move_count: int) -> Data:
        """
        מקודד מצב לוח נוכחי לאובייקט Data של torch_geometric.
        """
        node_features = np.zeros((self.num_nodes, self.num_node_features), dtype=np.float32)
        for pos, node_idx in self.node_map.items():
            is_peg = board.get(pos) == 1
            is_center = (pos == (3, 3))
            node_features[node_idx, 0] = is_peg
            node_features[node_idx, 1] = is_center
            # נרמול קואורדינטות
            node_features[node_idx, 2] = pos[0] / 6.0
            node_features[node_idx, 3] = pos[1] / 6.0

        # תכונות גלובליות שיצורפו לאובייקט
        pegs_left = float(board.count_pegs())

        return Data(
            x=torch.from_numpy(node_features),
            edge_index=self.edge_index,
            pegs_left=torch.tensor([[pegs_left / Board.TOTAL_PEGS]], dtype=torch.float32),
            move_count=torch.tensor([[move_count / Board.TOTAL_PEGS]], dtype=torch.float32)
        )

# =================================================================
# 3. שינוי מרכזי: ארכיטקטורת רשת עצבית גרפית (GNN)
# =================================================================
class PegSolitaireGNN(nn.Module):
    def __init__(self, node_features=4, num_actions=196, gnn_channels=64, num_heads=4):
        super(PegSolitaireGNN, self).__init__()
        # 2 global features: pegs_left, move_count
        global_features = 2

        # GNN Layers
        self.conv1 = GATv2Conv(node_features, gnn_channels, heads=num_heads, concat=True)
        self.conv2 = GATv2Conv(gnn_channels * num_heads, gnn_channels, heads=num_heads, concat=True)
        self.conv3 = GATv2Conv(gnn_channels * num_heads, gnn_channels, heads=1, concat=False)

        # Value Head
        self.value_fc1 = nn.Linear(gnn_channels + global_features, 64)
        self.value_fc2 = nn.Linear(64, 1)

        # Policy Head
        self.policy_fc1 = nn.Linear(gnn_channels + global_features, 64)
        self.policy_fc2 = nn.Linear(64, num_actions)

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GNN forward pass
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))

        # Global pooling to get a single vector for the whole graph
        graph_embedding = global_mean_pool(x, batch)

        # Concatenate global features (like peg count) to the graph embedding
        combined_embedding = torch.cat([graph_embedding, data.pegs_left, data.move_count], dim=1)

        # Value head
        v = F.relu(self.value_fc1(combined_embedding))
        v = torch.tanh(self.value_fc2(v))

        # Policy head
        p = F.relu(self.policy_fc1(combined_embedding))
        p_logits = self.policy_fc2(p)

        return p_logits, v

# =================================================================
# 4. התאמת סביבת המשחק והמאיץ
# =================================================================
class PegSolitaireEnv:
    def __init__(self, graph_encoder: GraphEncoder):
       self.game = Game()
       self.graph_encoder = graph_encoder
       self.done = False
       self.move_count = 0

    def reset(self):
       self.game = Game()
       self.done = False
       self.move_count = 0
       return self.get_observation()

    def step(self, action: Tuple[int, int, int]):
       peg_count_before = self.game.board.count_pegs()
       self.game.apply_action(action)
       peg_count_after = self.game.board.count_pegs()
       self.move_count += 1
       self.done = self.game.is_game_over()

       # Reward shaping: +1 for each peg removed, big bonus/penalty at the end
       reward = float(peg_count_before - peg_count_after)
       if self.done:
          reward += 100.0 if self.game.is_win() else -10.0

       return self.get_observation(), reward, self.done, {}

    def get_legal_moves(self):
       return self.game.get_legal_moves()

    def get_observation(self) -> Data:
        return self.graph_encoder.encode(self.game.board, self.move_count)

    def get_final_reward(self):
        pegs_left = self.game.board.count_pegs()
        if self.game.is_win():
            return float(Board.TOTAL_PEGS) + 100.0
        # Reward is higher for fewer pegs.
        return float(Board.TOTAL_PEGS - pegs_left)

    def clone(self):
       return copy.deepcopy(self)

# =================================================================
# 5. MCTS, Replay Buffer (עם התאמות קלות)
# =================================================================
class _Node:
    def __init__(self, prior: float):
       self.visit_count: int = 0
       self.value_sum: float = 0.0
       self.prior: float = prior
       self.children: Dict[int, "_Node"] = {}

    @property
    def value(self) -> float:
       return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

class MCTS:
    def __init__(self, game_env, model, action_space, config):
       self.game_env = game_env
       self.model = model
       self.action_space = action_space
       self.config = config
       self.device = config['device']

    def run(self, temperature: float = 1.0) -> np.ndarray:
       root = _Node(prior=0.0)
       current_obs = self.game_env.get_observation()
       legal_actions = self.game_env.get_legal_moves()
       if not legal_actions:
           return np.zeros(len(self.action_space), dtype=np.float32)

       self._expand(root, current_obs, legal_actions, add_noise=True)

       for _ in range(self.config['mcts_simulations']):
          game_copy = self.game_env.clone()
          node = root
          path = [root]
          search_path = [(None, root)]

          while node.children:
             action_idx, node = self._select_child(node)
             game_copy.step(self.action_space.from_index(action_idx))
             path.append(node)
             search_path.append((action_idx, node))

          value = 0.0
          if not game_copy.done:
             obs = game_copy.get_observation()
             legal_moves = game_copy.get_legal_moves()
             value = self._evaluate_state(obs)
             if legal_moves:
                self._expand(node, obs, legal_moves)
          else:
             value = game_copy.get_final_reward()

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

    def _expand(self, node: _Node, obs: Data, legal_actions: List, add_noise: bool = False):
       obs.to(self.device)
       with torch.no_grad():
          policy_logits, _ = self.model(obs)
          policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy().flatten()

       if add_noise and legal_actions:
          noise = np.random.dirichlet([self.config['dirichlet_alpha']] * len(legal_actions))
          for i, action in enumerate(legal_actions):
             action_idx = self.action_space.to_index(action)
             policy_probs[action_idx] = (1 - self.config['noise_fraction']) * policy_probs[action_idx] + self.config['noise_fraction'] * noise[i]

       for action in legal_actions:
          action_idx = self.action_space.to_index(action)
          if action_idx is not None:
             node.children[action_idx] = _Node(prior=policy_probs[action_idx])

    def _evaluate_state(self, obs: Data) -> float:
       obs.to(self.device)
       with torch.no_grad():
          _, value_tensor = self.model(obs)
          return value_tensor.item()

    def _calculate_final_policy(self, root: _Node, temperature: float) -> np.ndarray:
       if not root.children:
           return np.zeros(len(self.action_space), dtype=np.float32)
       visits = np.array([child.visit_count for child in root.children.values()], dtype=np.float32)
       action_indices = np.array(list(root.children.keys()))

       if temperature == 0:
          policy = np.zeros_like(visits)
          if len(visits) > 0:
             policy[np.argmax(visits)] = 1.0
       else:
          visits_temp = visits ** (1 / temperature)
          policy_sum = np.sum(visits_temp)
          policy = visits_temp / policy_sum if policy_sum > 0 else np.ones_like(visits) / len(visits)

       full_policy = np.zeros(len(self.action_space), dtype=np.float32)
       full_policy[action_indices] = policy
       return full_policy

class ReplayBuffer:
    def __init__(self, max_size: int):
       self.buffer: List[Tuple[Data, np.ndarray, float]] = []
       self.max_size = max_size

    def push(self, experience):
       if len(self.buffer) >= self.max_size:
          self.buffer.pop(0)
       self.buffer.append(experience)

    def sample(self, batch_size) -> List:
       indices = np.random.choice(len(self.buffer), batch_size, replace=False)
       return [self.buffer[i] for i in indices]

    def __len__(self):
       return len(self.buffer)

# =================================================================
# 6. מחלקת האימון (AlphaZeroTrainer) עם התאמה ל-GNN
# =================================================================
class AlphaZeroTrainer:
    def __init__(self, model, config):
       self.config = config
       self.device = config['device']
       self.model = model.to(self.device)
       self.action_space = Game.ACTION_SPACE
       self.graph_encoder = GraphEncoder()
       self.env = PegSolitaireEnv(self.graph_encoder)
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
          temp = self.config['temperature'] if self.env.move_count < self.config['temp_threshold'] else 0.0
          policy = self.mcts.run(temperature=temp)
          if np.sum(policy) == 0: break # No legal moves

          current_obs = self.env.get_observation()
          # We store the observation (Data object), the improved policy, and a placeholder for the final value
          game_history.append([current_obs, policy, 0.0])

          action_idx = np.random.choice(len(policy), p=policy)
          self.env.step(self.action_space.from_index(action_idx))

       final_reward = self.env.get_final_reward()
       for experience in game_history:
          experience[2] = final_reward # Update the placeholder with the actual final reward
       return game_history

    def _update_network(self):
       if len(self.replay_buffer) < self.config['batch_size']: return 0.0, 0.0
       self.model.train()

       batch = self.replay_buffer.sample(self.config['batch_size'])

       # Use torch_geometric's DataLoader to handle batching of graph data
       data_loader = DataLoader(batch, batch_size=self.config['batch_size'])
       graph_batch = next(iter(data_loader)).to(self.device)

       # The targets (policy, value) are part of the graph_batch object
       target_policies = graph_batch.y
       target_values = graph_batch.z.unsqueeze(1)

       pred_policies, pred_values = self.model(graph_batch)

       value_loss = self.value_loss_fn(pred_values, target_values)
       policy_loss = self.policy_loss_fn(pred_policies, target_policies)
       total_loss = value_loss + policy_loss

       self.optimizer.zero_grad()
       total_loss.backward()
       torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
       self.optimizer.step()
       return value_loss.item(), policy_loss.item()

    def train(self, num_iterations: int):
       for i in range(num_iterations):
          print(f"--- Iteration {i + 1}/{num_iterations} ---")
          self.model.eval()

          with tqdm(total=self.config['self_play_games_per_iter'], desc="Self-Playing") as pbar:
             for _ in range(self.config['self_play_games_per_iter']):
                game_data = self._self_play_game()
                # Augment and store data in the replay buffer
                for obs, pi, z_val in game_data:
                    # In a GNN, symmetry augmentation is more complex and often omitted.
                    # For now, we add the original data.
                    # We need to attach the targets to the Data object for the DataLoader
                    obs.y = torch.tensor(pi, dtype=torch.float32)
                    obs.z = torch.tensor(z_val, dtype=torch.float32)
                    self.replay_buffer.push(obs)
                pbar.update(1)

          total_v_loss, total_p_loss = 0, 0
          with tqdm(total=self.config['training_steps_per_iter'], desc="Training Network") as pbar:
             for _ in range(self.config['training_steps_per_iter']):
                v_loss, p_loss = self._update_network()
                total_v_loss += v_loss
                total_p_loss += p_loss
                pbar.set_postfix({"v_loss": f"{v_loss:.3f}", "p_loss": f"{p_loss:.3f}"})
                pbar.update(1)

          if (i + 1) % self.config['save_interval'] == 0:
             torch.save(self.model.state_dict(), f"peg_solitaire_gnn_model_iter_{i + 1}.pth")
          avg_v_loss = total_v_loss / self.config['training_steps_per_iter']
          avg_p_loss = total_p_loss / self.config['training_steps_per_iter']
          print(f"Buffer size: {len(self.replay_buffer)}, Avg Loss (V/P): {avg_v_loss:.4f}/{avg_p_loss:.4f}")

# =================================================================
# 7. הרצה ראשית
# =================================================================
if __name__ == '__main__':
    config = {
       'device': 'cuda' if torch.cuda.is_available() else 'cpu',
       'learning_rate': 0.001,
       'weight_decay': 1e-5,
       'buffer_size': 20_000,
       'batch_size': 128,
       'grad_clip': 1.0,
       'temperature': 1.0,
       'temp_threshold': 10,
       'self_play_games_per_iter': 50,
       'training_steps_per_iter': 100,
       'save_interval': 5,
       'mcts_simulations': 50, # Reduced for faster iterations with GNN
       'c_puct': 2.5,
       'dirichlet_alpha': 0.3,
       'noise_fraction': 0.25,
    }
    print(f"Using device: {config['device']}")

    model = PegSolitaireGNN(num_actions=Game.ACTION_SPACE.num_actions)
    trainer = AlphaZeroTrainer(model, config)

    print("Starting AlphaZero training with GNN for Peg Solitaire...")
    trainer.train(num_iterations=200)
    print("Training finished.")