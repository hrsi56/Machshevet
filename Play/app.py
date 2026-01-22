import os
import pickle
import numpy as np
from numba import njit, int64
from flask import Flask, render_template, jsonify, request


# ==========================================
#  חלק 0: האצת Numba (מועתק מהקוד המקורי)
# ==========================================
@njit(int64(int64, int64[:, :, :]))
def fast_canonical_lookup(board, table):
	min_val = board
	c0 = board & 0x7FF
	c1 = (board >> 11) & 0x7FF
	c2 = (board >> 22) & 0x7FF

	for sym_idx in range(8):
		mapped_board = (table[sym_idx, 0, c0] |
		                table[sym_idx, 1, c1] |
		                table[sym_idx, 2, c2])

		if mapped_board < min_val:
			min_val = mapped_board

	return min_val


# ==========================================
#  חלק 1: המנוע (Solver - ללא UI)
# ==========================================
class PegSolitaireSolver:
	MEMORY_FILE = "../solitaire_pro_brain.pkl"

	def __init__(self):
		self.r_c_to_bit = {}
		self.bit_to_r_c = {}
		self.valid_mask = 0
		self.center_bit = 0

		# מיפוי הלוח
		idx = 0
		for r in range(7):
			for c in range(7):
				if not ((r < 2 and c < 2) or (r < 2 and c > 4) or
				        (r > 4 and c < 2) or (r > 4 and c > 4)):
					bit = 1 << idx
					self.r_c_to_bit[(r, c)] = bit
					self.bit_to_r_c[bit] = (r, c)
					self.valid_mask |= bit
					if r == 3 and c == 3:
						self.center_bit = bit
					idx += 1

		# טעינת המוח
		self.brain = {}
		self.symmetry_table = None
		self.load_brain()

		# חישוב מהלכים אפשריים מראש
		self.moves_map = []
		self._precompute_moves()

	def load_brain(self):
		if os.path.exists(self.MEMORY_FILE):
			try:
				with open(self.MEMORY_FILE, "rb") as f:
					data = pickle.load(f)
					self.brain = data["brain"]
					self.symmetry_table = data["symmetry_table"]
			except Exception as e:
				print(f"Error loading brain: {e}")
		else:
			print("Brain file not found!")

	def _precompute_moves(self):
		directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
		for r in range(7):
			for c in range(7):
				if (r, c) in self.r_c_to_bit:
					src_bit = self.r_c_to_bit[(r, c)]
					for dr, dc in directions:
						mid_r, mid_c = r + dr, c + dc
						dst_r, dst_c = r + 2 * dr, c + 2 * dc
						if (mid_r, mid_c) in self.r_c_to_bit and (dst_r, dst_c) in self.r_c_to_bit:
							mid_bit = self.r_c_to_bit[(mid_r, mid_c)]
							dst_bit = self.r_c_to_bit[(dst_r, dst_c)]
							mask = src_bit | mid_bit | dst_bit
							# (mask, src, dst, mid)
							self.moves_map.append((mask, src_bit, dst_bit, mid_bit))

	def get_winning_moves_count(self, board):
		# לוגיקה פשוטה לבדיקת מצב (אם קיים בזיכרון)
		canon = fast_canonical_lookup(board, self.symmetry_table)
		if canon in self.brain:
			res = self.brain[canon]
			# הטיפול כאן פשוט יותר מהקוד המקורי לצורך ה-Web
			if res == 1: return "Win"  # True
			if res == 0: return "Loss"  # False
		return "Unknown"

	def solve_full_path(self, start_board):
		# חיפוש פתרון פשוט (BFS/DFS) על בסיס הזיכרון
		# לצורך הדוגמה נשתמש בלוגיקה הבסיסית של הקוד המקורי אם אפשר
		# אבל כאן נחזיר רשימה של מצבי לוח

		path = [start_board]
		curr = start_board

		# הגבלה למניעת לולאה אינסופית בדמו
		for _ in range(32):
			if curr == self.center_bit:
				return path  # נצחון

			found_move = False
			# נסה למצוא מהלך שמוביל למצב מנצח
			for mask, src, dst, mid in self.moves_map:
				# אם המהלך חוקי: יש מקור, יש אמצע, אין יעד
				if (curr & mask) == (src | mid):
					next_state = curr ^ mask
					canon = fast_canonical_lookup(next_state, self.symmetry_table)
					if self.brain.get(canon) == 1:  # זה נתיב מנצח
						curr = next_state
						path.append(curr)
						found_move = True
						break

			if not found_move:
				return None  # אין פתרון מכאן ע"פ הזיכרון

		return path


# ==========================================
#  Flask Web App
# ==========================================
app = Flask(__name__)
solver = PegSolitaireSolver()


@app.route('/')
def index():
	return render_template('index.html')


@app.route('/api/init')
def init_game():
	# לוח מלא חוץ מהאמצע
	full_board = solver.valid_mask
	start_board = full_board ^ solver.center_bit
	return jsonify({'board': int(start_board)})


@app.route('/api/check_status', methods=['POST'])
def check_status():
	data = request.json
	board = int(data.get('board', 0))
	status = solver.get_winning_moves_count(board)
	return jsonify({'status': status})


@app.route('/api/solve', methods=['POST'])
def solve():
	data = request.json
	board = int(data.get('board', 0))
	path = solver.solve_full_path(board)
	if path:
		# המרת כל שלב בפתרון למספר שלם
		return jsonify({'path': [int(s) for s in path]})
	else:
		return jsonify({'error': 'No solution found'})


if __name__ == '__main__':
	# הרצה מקומית, ניתן לגשת מהנייד אם באותה רשת WiFi
	app.run(host='0.0.0.0', port=5000, debug=True)