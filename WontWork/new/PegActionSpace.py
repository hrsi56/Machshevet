import numpy as np
from typing import List, Tuple, Dict


class PegActionSpace:
	"""
	מנהל את מרחב הפעולות עבור פג סוליטר 7x7.
	מתרגם בין פעולות (שורה, עמודה, כיוון) לאינדקסים מספריים.
	"""

	def __init__(self, board_size: int = 7):
		self.board_size = board_size
		self.directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]  # 0:Up, 1:Down, 2:Left, 3:Right
		self.num_directions = len(self.directions)
		self.num_actions = self.board_size * self.board_size * self.num_directions

		# --- יצירת מנגנון תרגום מהיר ---
		# יצירת מילונים לתרגום מהיר במקום חישובים חוזרים
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
		"""
		ממיר פעולה (שורה, עמודה, אינדקס כיוון) לאינדקס מספרי.
		לדוגמה: to_index((3, 3, 0)) -> 140
		"""
		return self._action_to_index.get(action)

	def from_index(self, index: int) -> Tuple[int, int, int]:
		"""
		ממיר אינדקס מספרי בחזרה לפעולה (שורה, עמודה, אינדקס כיוון).
		לדוגמה: from_index(140) -> (3, 3, 0)
		"""
		return self._index_to_action[index]

	def action_to_move(self, action: Tuple[int, int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
		"""
		מתרגם פעולה קנונית למהלך של (from_pos, to_pos) שהמשחק מבין.
		"""
		from_r, from_c, dir_idx = action
		dr, dc = self.directions[dir_idx]
		to_pos = (from_r + dr, from_c + dc)
		from_pos = (from_r, from_c)
		return from_pos, to_pos

	def __len__(self) -> int:
		""" מחזיר את הגודל הכולל של מרחב הפעולות. """
		return self.num_actions

	def __getitem__(self, index: int) -> Tuple[int, int, int]:
		""" מאפשר גישה נוחה כמו ברשימה: action_space[140] """
		return self.from_index(index)