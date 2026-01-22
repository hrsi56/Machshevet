from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # ספרייה נחמדה להצגת פסי התקדמות
import MCTS
from NET import ReplayBuffer


class AlphaZeroTrainer:
	"""
	המחלקה המרכזית שמנהלת את לולאת האימון של AlphaZero.
	"""

	def __init__(self, model, game_env, mcts_simulations, action_space, config):
		self.model = model
		self.game_env = game_env
		self.action_space = action_space
		self.config = config
		self.device = config['device']

		# יצירת מופע של MCTS עם הרשת הנוכחית
		self.mcts = MCTS(
			game_env=self.game_env,
			model=self.model,
			action_space=self.action_space,
			simulations=mcts_simulations,
			device=self.device
		)

		# מאגר הזיכרון Prioritized Experience Replay
		self.replay_buffer = ReplayBuffer(max_size=config['buffer_size'])

		# אופטימייזר לאימון הרשת
		self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'],
		                            weight_decay=config['weight_decay'])

		# פונקציות הפסד (Loss functions)
		self.value_loss_fn = nn.MSELoss()
		self.policy_loss_fn = nn.CrossEntropyLoss()

	def _self_play_game(self) -> List[Tuple]:
		"""
		מריץ משחק בודד מההתחלה ועד הסוף, ואוסף דאטה לאימון.
		"""
		game_history = []
		self.game_env.reset()

		while not self.game_env.done:
			# הרצת MCTS כדי לקבל את וקטור המדיניות המשופר (π)
			policy = self.mcts.run(temperature=self.config['temperature'])

			# שמירת המצב הנוכחי והמדיניות שחושבה
			current_state_obs = self.game_env.encode_observation()
			game_history.append([current_state_obs, policy, 0.0])  # ה-0.0 הוא placeholder לערך הסופי

			# בחירת הפעולה הבאה באופן הסתברותי לפי המדיניות
			action_idx = np.random.choice(len(policy), p=policy)
			action = self.action_space[action_idx]  # המרת אינדקס לפעולה

			# ביצוע המהלך בסביבה
			self.game_env.step(action)

		# --- המשחק הסתיים, עדכון הערך האמיתי (z) לאחור ---
		final_reward = self.game_env.get_final_reward(is_win=self.game_env.game.is_win())

		for i in range(len(game_history)):
			# הערך של כל מצב הוא התוצאה הסופית של המשחק
			game_history[i][2] = final_reward

		return game_history

	def _update_network(self) -> Tuple[float, float]:
		"""
		דוגם batch ממאגר הזיכרון ומבצע צעד אימון בודד על הרשת.
		"""
		if len(self.replay_buffer) < self.config['batch_size']:
			return 0.0, 0.0  # לא מאמנים אם אין מספיק דגימות

		self.model.train()  # מעבר למצב אימון

		# דגימת batch ממאגר הזיכרון (עם עדיפות)
		states, target_policies, target_values, sample_indices = self.replay_buffer.sample_as_tensors(
			self.config['batch_size'], self.device
		)

		# קבלת הניבויים מהרשת
		pred_policies, pred_values = self.model(states)

		# חישוב ה-Loss
		value_loss = self.value_loss_fn(pred_values.squeeze(), target_values)
		policy_loss = self.policy_loss_fn(pred_policies, target_policies)
		total_loss = value_loss + policy_loss

		# עדכון משקולות הרשת
		self.optimizer.zero_grad()
		total_loss.backward()
		self.optimizer.step()

		# עדכון העדיפויות במאגר הזיכרון (עבור PER)
		td_error = torch.abs(pred_values.squeeze() - target_values).detach().cpu().numpy()
		self.replay_buffer.update_priorities(sample_indices, td_error)

		return value_loss.item(), policy_loss.item()

	def train_loop(self, num_iterations: int):
		"""
		לולאת האימון הראשית.
		"""
		for i in range(num_iterations):
			print(f"--- Iteration {i + 1}/{num_iterations} ---")

			# --- שלב 1: יצירת דאטה ממשחק עצמי ---
			self.model.eval()  # מעבר למצב הערכה (חשוב לרעש ועקביות)
			new_game_data = []
			num_games = self.config['self_play_games_per_iter']

			for _ in tqdm(range(num_games), desc="Self-Playing"):
				new_game_data.extend(self._self_play_game())

			# הוספת הדאטה החדש למאגר הזיכרון
			for sample in new_game_data:
				self.replay_buffer.push(sample)

			# --- שלב 2: אימון הרשת ---
			num_training_steps = self.config['training_steps_per_iter']
			total_v_loss, total_p_loss = 0, 0

			for _ in tqdm(range(num_training_steps), desc="Training Network"):
				v_loss, p_loss = self._update_network()
				total_v_loss += v_loss
				total_p_loss += p_loss

			avg_v_loss = total_v_loss / num_training_steps if num_training_steps > 0 else 0
			avg_p_loss = total_p_loss / num_training_steps if num_training_steps > 0 else 0

			print(f"Iteration {i + 1} complete. Avg Value Loss: {avg_v_loss:.4f}, Avg Policy Loss: {avg_p_loss:.4f}")

			# אפשר להוסיף פה שמירה של המודל כל כמה איטרציות
			if (i + 1) % self.config['save_interval'] == 0:
				torch.save(self.model.state_dict(), f"model_iter_{i + 1}.pth")