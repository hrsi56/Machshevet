import numpy as np
import torch
from typing import Dict, List, Tuple

class _Node:
    """ מייצג צומת בודד בעץ החיפוש של MCTS. """
    def __init__(self, prior: float):
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.prior: float = prior  # הסתברות P(s,a) מהרשת
        self.children: Dict[int, "_Node"] = {}

    @property
    def value(self) -> float:
        """ מחשב את הערך הממוצע של הצומת (Q-value). """
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count



class MCTS:
    """
    PUCT-based Monte-Carlo Tree Search, AlphaZero style.
    מימוש מלא המשלב רעש דיריכלה, עצירה דינמית, ואינטגרציה
    עקבית עם סביבת המשחק.
    """

    def __init__(
            self,
            game_env,  # מופע של המחלקה Game שלנו
            model: torch.nn.Module,
            action_space,
            simulations: int = 100,
            c_puct: float = 1.5,
            dirichlet_alpha: float = 0.3,
            noise_fraction: float = 0.25,
            device: str = "cpu",
    ):
        self.game_env = game_env
        self.model = model
        self.action_space = action_space
        self.simulations = simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_fraction = noise_fraction
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def run(self, temperature: float = 1.0, entropy_threshold: float = 0.5) -> np.ndarray:
        """
        מריץ את אלגוריתם MCTS מהמצב הנוכחי של סביבת המשחק.
        """
        # --- 1. הכנת השורש ---
        root = _Node(prior=0.0)
        current_obs = self.game_env.board.as_array()  # נניח שיש לנו מתודה כזו
        legal_actions = self.game_env.get_legal_moves()
        self._expand(root, current_obs, legal_actions, add_noise=True)

        # --- 2. לולאת הסימולציות ---
        min_sims = int(self.simulations * 0.3)
        for i in range(self.simulations):
            game_copy = self.game_env.clone()  # שכפול הסביבה לכל סימולציה
            node = root
            path = [root]
            done = False

            # --- שלב א': בחירה (Selection) ---
            while node.children:
                action_idx, node = self._select_child(node)
                path.append(node)
                # עדכון מצב המשחק המשוכפל
                _, _, done, _ = game_copy.apply_move(*self.action_space[action_idx])

            # --- שלב ב': הרחבה (Expansion) והערכה (Evaluation) ---
            value = 0.0
            if not done:
                legal_actions = game_copy.get_legal_moves()
                obs = game_copy.board.as_array()
                self._expand(node, obs, legal_actions)
                value = self._evaluate_state(obs)
            else:
                # הגדרת ערך סופי עקבי עם פונקציית התגמול שלנו
                value = game_copy.get_final_reward(is_win=game_copy.is_win())

            # --- שלב ג': התפשטות לאחור (Backpropagation) ---
            for node_in_path in reversed(path):
                node_in_path.visit_count += 1
                node_in_path.value_sum += value

            # --- אופטימיזציה: עצירה דינמית ---
            if i > min_sims:
                visits = np.array([child.visit_count for child in root.children.values()])
                if visits.sum() > 0:
                    probs = visits / visits.sum()
                    entropy = -np.sum(probs * np.log(probs + 1e-9))
                    if entropy < entropy_threshold:
                        break  # עצירה מוקדמת אם העץ "בטוח"

        # --- 3. חישוב מדיניות הפעולה הסופית (π) ---
        return self._calculate_final_policy(root, temperature)

    def _select_child(self, node: _Node) -> Tuple[int, _Node]:
        """ בוחר את הילד עם ציון ה-PUCT הגבוה ביותר. """
        sqrt_total_visits = np.sqrt(node.visit_count)
        best_score, best_action_idx, best_child = -np.inf, -1, None

        for action_idx, child_node in node.children.items():
            q_value = child_node.value
            u_value = self.c_puct * child_node.prior * sqrt_total_visits / (1 + child_node.visit_count)
            score = q_value + u_value
            if score > best_score:
                best_score = score
                best_action_idx = action_idx
                best_child = child_node

        return best_action_idx, best_child

    def _expand(self, node: "_Node", obs: np.ndarray, legal_actions: List[Tuple], add_noise: bool = False):
        """ מרחיב צומת עלה: מקבל מהרשת הסתברויות ומאכלס את הילדים של הצומת. """
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            policy_logits, _ = self.model(obs_tensor)
            policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy().flatten()

        # החלת רעש דיריכלה (רק על השורש)
        if add_noise:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_actions))
            for i, action in enumerate(legal_actions):
                action_idx = self.action_space.index(action)
                policy_probs[action_idx] = (1 - self.noise_fraction) * policy_probs[action_idx] + self.noise_fraction * \
                                           noise[i]

        # יצירת צמתי הילדים עבור כל מהלך חוקי
        for action in legal_actions:
            action_idx = self.action_space.to_index(action)
            node.children[action_idx] = _Node(prior=float(policy_probs[action_idx]))

    def _evaluate_state(self, obs: np.ndarray) -> float:
        """ מעריך את הערך (V) של מצב נתון באמצעות הרשת. """
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, value_tensor = self.model(obs_tensor)
            return value_tensor.item()

    def _calculate_final_policy(self, root: _Node, temperature: float) -> np.ndarray:
        """ מחשב את וקטור ההסתברויות הסופי לבחירת מהלך. """
        visits = np.array([child.visit_count for child in root.children.values()])
        action_indices = np.array(list(root.children.keys()))

        if temperature == 0:  # בחירה דטרמיניסטית
            policy = np.zeros_like(visits, dtype=float)
            policy[np.argmax(visits)] = 1.0
        else:  # בחירה הסתברותית
            visits_temp = visits ** (1 / temperature)
            policy = visits_temp / visits_temp.sum()

        full_policy = np.zeros(len(self.action_space), dtype=float)
        full_policy[action_indices] = policy
        return full_policy
