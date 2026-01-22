import tkinter as tk
from tkinter import messagebox
import threading
import pickle
import os
import time
from collections import deque
import numpy as np
from numba import njit, int64


# ==========================================
#  חלק 0: האצת Numba (מחוץ למחלקה)
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
#  חלק 1: המנוע (Solver)
# ==========================================
class PegSolitaireSolver:
	MEMORY_FILE = "solitaire_pro_brain.pkl"

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
					self.r_c_to_bit[(r, c)] = idx
					self.bit_to_r_c[idx] = (r, c)
					self.valid_mask |= (1 << idx)
					if r == 3 and c == 3:
						self.center_bit = idx
					idx += 1

		self.symmetry_maps = self._generate_base_symmetry_maps()
		self.lookup_table = self._build_numpy_lookup_table()

		# הכנת מהלכים
		self.moves = []
		directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
		for r in range(7):
			for c in range(7):
				if (r, c) not in self.r_c_to_bit: continue
				src = self.r_c_to_bit[(r, c)]
				for dr, dc in directions:
					mr, mc = r + dr // 2, c + dc // 2
					drr, dcc = r + dr, c + dc
					if (mr, mc) in self.r_c_to_bit and (drr, dcc) in self.r_c_to_bit:
						mid = self.r_c_to_bit[(mr, mc)]
						dst = self.r_c_to_bit[(drr, dcc)]
						mask = (1 << src) | (1 << mid) | (1 << dst)
						self.moves.append({
							'mask': mask,
							'check_src': (1 << src) | (1 << mid),
							'check_dst': (1 << dst),
							'src': src, 'dst': dst
						})

		self.winning_states = set()
		if not self.load_memory():
			raise FileNotFoundError(f"Missing Brain File: {self.MEMORY_FILE}")

	def _generate_base_symmetry_maps(self):
		maps = []
		for i in range(8):
			mapping = {}
			for r in range(7):
				for c in range(7):
					if (r, c) not in self.r_c_to_bit: continue
					rr, cc = r, c
					if i & 4: rr, cc = cc, rr
					if i & 1: rr, cc = cc, 6 - rr
					if i & 2: rr, cc = 6 - rr, 6 - cc
					src_idx = self.r_c_to_bit[(r, c)]
					dst_idx = self.r_c_to_bit[(rr, cc)]
					mapping[src_idx] = dst_idx
			maps.append(mapping)
		return maps

	def _build_numpy_lookup_table(self):
		table = np.zeros((8, 3, 2048), dtype=np.int64)
		for sym_idx in range(8):
			mapping = self.symmetry_maps[sym_idx]
			for chunk_id in range(3):
				bit_offset = chunk_id * 11
				for val in range(2048):
					transformed_val = 0
					t = val
					while t:
						lsb = t & -t
						local_idx = lsb.bit_length() - 1
						real_idx = local_idx + bit_offset
						if real_idx in mapping:
							target_idx = mapping[real_idx]
							transformed_val |= (1 << target_idx)
						t ^= lsb
					table[sym_idx, chunk_id, val] = transformed_val
		return table

	def get_canonical(self, board):
		return fast_canonical_lookup(board, self.lookup_table)

	def get_initial_board(self):
		board = self.valid_mask
		board &= ~(1 << self.center_bit)
		return board

	def load_memory(self):
		if os.path.exists(self.MEMORY_FILE):
			try:
				with open(self.MEMORY_FILE, "rb") as f:
					self.winning_states = pickle.load(f)
				return True
			except Exception:
				return False
		return False

	def get_winning_moves_count(self, board):
		""" מחזיר כמה מהלכים מנצחים קיימים מהמצב הנתון """
		# אם הלוח הנוכחי הוא כבר לא במצב מנצח (לא ב-DB), אין טעם לבדוק
		if self.get_canonical(board) not in self.winning_states and board != (1 << self.center_bit):
			return 0

		count = 0
		for m in self.moves:
			if (board & m['check_src'] == m['check_src']) and (board & m['check_dst'] == 0):
				next_board = board ^ m['mask']
				if self.get_canonical(next_board) in self.winning_states or next_board == (1 << self.center_bit):
					count += 1
		return count

	def solve_full_path(self, start_board):
		path = []
		if self._find_path_forward(start_board, path):
			return self._reconstruct(start_board, path)
		return None

	def _find_path_forward(self, board, path):
		if board == (1 << self.center_bit): return True
		if self.get_canonical(board) not in self.winning_states: return False

		candidates = []
		for m in self.moves:
			if (board & m['check_src'] == m['check_src']) and (board & m['check_dst'] == 0):
				next_board = board ^ m['mask']
				if self.get_canonical(next_board) in self.winning_states:
					candidates.append((m, next_board))

		candidates.sort(key=lambda x: self.get_winning_moves_count(x[1]), reverse=True)

		for m, next_b in candidates:
			path.append(m)
			if self._find_path_forward(next_b, path):
				return True
			path.pop()
		return False

	def _reconstruct(self, start, moves):
		states = [start]
		curr = start
		for m in moves:
			curr ^= m['mask']
			states.append(curr)
		return states


# ==========================================
#  חלק 2: רכיב הגרף (GUI)
# ==========================================
class SurvivalFunnelGraph(tk.Canvas):
	def __init__(self, parent, width=300, height=120, bg="#222"):
		super().__init__(parent, width=width, height=height, bg=bg, highlightthickness=0)
		self.w, self.h = width, height
		self.data_points = []
		self.max_val = 10
		self.create_line(0, self.h, self.w, self.h, fill="#444", width=2)

	def reset(self):
		self.data_points = []
		self.delete("graph")

	def update_graph(self, val):
		self.data_points.append(val)
		if val > self.max_val: self.max_val = val
		self.draw()

	def undo(self):
		if self.data_points:
			self.data_points.pop()
			self.draw()

	def draw(self):
		self.delete("graph")
		self.delete("text")
		if not self.data_points: return

		curr = self.data_points[-1]
		color = "#00E676"
		if curr == 0:
			color = "#D50000"
		elif curr == 1:
			color = "#FF9100"

		step = self.w / max(32, len(self.data_points))
		coords = []
		for i, val in enumerate(self.data_points):
			x = i * step
			h_ratio = min(1.0, val / max(1, self.max_val))
			y = self.h - (h_ratio * (self.h - 20)) - 10
			coords.extend([x, y])

		if len(coords) >= 4:
			self.create_line(*coords, fill=color, width=3, smooth=True, tag="graph")
			poly = coords[:] + [coords[-2], self.h, 0, self.h]
			self.create_polygon(*poly, fill=color, stipple="gray25", outline="", tag="graph")

		cx, cy = coords[-2], coords[-1]
		self.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill="white", outline=color, width=2, tag="graph")

		msg = f"Winning Moves: {curr}"
		if curr == 0: msg = "DEAD END"
		self.create_text(10, 15, text=msg, anchor="w", fill="white", font=("Consolas", 10, "bold"), tag="text")


# ==========================================
#  חלק 3: ממשק המשתמש (GUI) - עם Heatmap
# ==========================================
class PegSolitaireGUI:
	CELL_SIZE = 45
	PADDING = 10
	COLOR_BG = "#1e1e1e"
	COLOR_HOLE = "#333333"
	COLOR_PEG = "#FFD700"
	COLOR_SELECTED = "#2962FF"

	# צבעי ה-Heatmap
	COLOR_HEAT_GOOD = "#00C853"  # ירוק - מהלך טוב
	COLOR_HEAT_BAD = "#D50000"  # אדום - מבוי סתום

	def __init__(self, root, solver):
		self.root = root
		self.solver = solver
		self.root.title("Solitaire Pro 🚀 (Heatmap Enabled)")
		self.root.configure(bg=self.COLOR_BG)
		self.current_board = self.solver.get_initial_board()
		self.selected = None
		self.history = []
		self.animating = False
		self._init_ui()
		self.draw_board()

		initial_moves = self.solver.get_winning_moves_count(self.current_board)
		self.graph.update_graph(initial_moves)

	def _init_ui(self):
		main = tk.Frame(self.root, bg=self.COLOR_BG)
		main.pack(fill="both", expand=True, padx=20, pady=20)

		# Left Panel
		left = tk.Frame(main, bg=self.COLOR_BG)
		left.pack(side="left")
		sz = 7 * self.CELL_SIZE + 2 * self.PADDING
		self.cvs = tk.Canvas(left, width=sz, height=sz, bg=self.COLOR_BG, highlightthickness=0)
		self.cvs.pack()
		self.cvs.bind("<Button-1>", self.on_click)

		btns = tk.Frame(left, bg=self.COLOR_BG)
		btns.pack(pady=10)
		tk.Button(btns, text="Undo", command=self.undo, width=8).pack(side=tk.LEFT, padx=5)
		tk.Button(btns, text="Auto Solve", command=self.auto_solve, bg="#00C853", fg="white").pack(side=tk.LEFT, padx=5)
		tk.Button(btns, text="Reset", command=self.reset, width=8).pack(side=tk.LEFT, padx=5)

		# Right Panel
		right = tk.Frame(main, bg=self.COLOR_BG)
		right.pack(side="right", fill="y", padx=(20, 0))
		tk.Label(right, text="Survival Funnel", font=("Segoe UI", 12, "bold"), fg="#aaa", bg=self.COLOR_BG).pack(
			anchor="w")
		self.graph = SurvivalFunnelGraph(right, width=250, height=150)
		self.graph.pack(pady=(5, 20))
		self.lbl_status = tk.Label(right, text="Game Start", font=("Segoe UI", 16, "bold"), fg="white",
		                           bg=self.COLOR_BG)
		self.lbl_status.pack(pady=10)

		# מקרא
		tk.Label(right, text="Heatmap Legend:", font=("Segoe UI", 10, "bold"), fg="#aaa", bg=self.COLOR_BG).pack(
			anchor="w", pady=(10, 0))
		legend_frame = tk.Frame(right, bg=self.COLOR_BG)
		legend_frame.pack(anchor="w", pady=5)
		tk.Label(legend_frame, text="■", fg=self.COLOR_HEAT_GOOD, bg=self.COLOR_BG).pack(side=tk.LEFT)
		tk.Label(legend_frame, text=" Safe Move (# futures)", fg="#ccc", bg=self.COLOR_BG, font=("Segoe UI", 9)).pack(
			side=tk.LEFT, padx=(0, 10))
		tk.Label(legend_frame, text="■", fg=self.COLOR_HEAT_BAD, bg=self.COLOR_BG).pack(side=tk.LEFT)
		tk.Label(legend_frame, text=" Dead End (0)", fg="#ccc", bg=self.COLOR_BG, font=("Segoe UI", 9)).pack(
			side=tk.LEFT)

	def draw_board(self):
		self.cvs.delete("all")

		# 1. ציור הלוח הבסיסי
		for r in range(7):
			for c in range(7):
				if (r, c) not in self.solver.r_c_to_bit: continue
				idx = self.solver.r_c_to_bit[(r, c)]
				has_peg = (self.current_board >> idx) & 1
				x = self.PADDING + c * self.CELL_SIZE + self.CELL_SIZE // 2
				y = self.PADDING + r * self.CELL_SIZE + self.CELL_SIZE // 2
				fill = self.COLOR_PEG if has_peg else self.COLOR_HOLE
				out = ""
				width = 1
				if has_peg and self.selected == (r, c):
					fill = self.COLOR_SELECTED;
					out = "white";
					width = 2
				self.cvs.create_oval(x - 18, y - 18, x + 18, y + 18, fill=fill, outline=out, width=width)

		# 2. ציור ה-HEATMAP אם יש בחירה
		if self.selected and not self.animating:
			sr, sc = self.selected
			s_idx = self.solver.r_c_to_bit[(sr, sc)]

			# עוברים על כל המהלכים האפשריים מהכלי שנבחר
			for m in self.solver.moves:
				if m['src'] == s_idx:
					# בדיקת חוקיות בסיסית (יש אמצע, יעד ריק)
					if (self.current_board & m['check_src'] == m['check_src']) and \
							(self.current_board & m['check_dst'] == 0):

						# חיזוי העתיד: מה יקרה אם אבחר במהלך הזה?
						next_board = self.current_board ^ m['mask']

						# בדיקה ב"מוח": האם המצב הבא הוא מצב מנצח?
						is_winning_path = False
						future_moves = 0

						if next_board == (1 << self.solver.center_bit):
							# ניצחון מיידי
							is_winning_path = True
							future_moves = 99
						elif self.solver.get_canonical(next_board) in self.solver.winning_states:
							is_winning_path = True
							future_moves = self.solver.get_winning_moves_count(next_board)

						# ציור האינדיקטור
						dst_r, dst_c = self.solver.bit_to_r_c[m['dst']]
						self._draw_heat_indicator(dst_r, dst_c, future_moves, is_winning_path)

	def _draw_heat_indicator(self, r, c, count, is_safe):
		x = self.PADDING + c * self.CELL_SIZE + self.CELL_SIZE // 2
		y = self.PADDING + r * self.CELL_SIZE + self.CELL_SIZE // 2

		color = self.COLOR_HEAT_GOOD if is_safe else self.COLOR_HEAT_BAD
		text = str(count)

		# עיגול רקע קטן
		self.cvs.create_oval(x - 12, y - 12, x + 12, y + 12, fill=color, outline="white", width=1)
		# טקסט המספר
		self.cvs.create_text(x, y, text=text, fill="white", font=("Segoe UI", 9, "bold"))

	def on_click(self, e):
		if self.animating: return
		c, r = (e.x - self.PADDING) // self.CELL_SIZE, (e.y - self.PADDING) // self.CELL_SIZE
		if (r, c) not in self.solver.r_c_to_bit: return
		idx = self.solver.r_c_to_bit[(r, c)]
		has_peg = (self.current_board >> idx) & 1

		if self.selected:
			if self.selected == (r, c):
				self.selected = None
			elif has_peg:
				# שינוי בחירה לחייל אחר
				self.selected = (r, c)
			else:
				# ניסיון תזוזה לחור ריק
				self.try_move(self.selected, (r, c))
		elif has_peg:
			self.selected = (r, c)

		self.draw_board()

	def try_move(self, src, dst):
		s_idx = self.solver.r_c_to_bit[src]
		d_idx = self.solver.r_c_to_bit[dst]
		move = None
		for m in self.solver.moves:
			if m['src'] == s_idx and m['dst'] == d_idx:
				if (self.current_board & (m['check_src'] ^ (1 << s_idx))): move = m
				break
		if move:
			self.history.append(self.current_board)
			self.current_board ^= move['mask']
			self.selected = None
			wins = self.solver.get_winning_moves_count(self.current_board)
			self.graph.update_graph(wins)
			self.draw_board()
			if wins > 0:
				self.lbl_status.config(text="Safe Move ✅", fg="#00E676")
			elif self.current_board == (1 << self.solver.center_bit):
				self.lbl_status.config(text="VICTORY! 🏆", fg="gold")
			else:
				self.lbl_status.config(text="Dead End ❌", fg="#D50000")
		else:
			# אם לחץ על מקום לא חוקי, רק נבטל בחירה
			self.selected = None
			self.draw_board()

	def undo(self):
		if self.history:
			self.current_board = self.history.pop()
			self.selected = None
			self.graph.undo()
			self.draw_board()
			self.lbl_status.config(text="Undo", fg="white")

	def reset(self):
		self.history = []
		self.current_board = self.solver.get_initial_board()
		self.graph.reset()
		initial_moves = self.solver.get_winning_moves_count(self.current_board)
		self.graph.update_graph(initial_moves)
		self.draw_board()
		self.lbl_status.config(text="Game Start", fg="white")

	def auto_solve(self):
		if self.animating: return
		threading.Thread(target=self._solve_thread, daemon=True).start()

	def _solve_thread(self):
		path = self.solver.solve_full_path(self.current_board)
		if path:
			self.root.after(0, lambda: self._animate(path))
		else:
			self.root.after(0, lambda: messagebox.showinfo("Info", "Cannot solve from here"))

	def _animate(self, states, idx=0):
		if idx >= len(states): self.animating = False; return
		self.animating = True
		self.current_board = states[idx]
		wins = self.solver.get_winning_moves_count(states[idx])
		self.graph.update_graph(wins)
		self.draw_board()
		self.root.after(200, lambda: self._animate(states, idx + 1))


# ==========================================
#  Main Entry Point
# ==========================================
def main():
	root = tk.Tk()
	root.withdraw()

	try:
		solver = PegSolitaireSolver()
		root.deiconify()
		root.geometry("650x550")
		PegSolitaireGUI(root, solver)
		root.mainloop()

	except FileNotFoundError:
		messagebox.showerror("Error",
		                     f"Brain file not found!\n\nPlease place '{PegSolitaireSolver.MEMORY_FILE}' in the folder.")
		root.destroy()
	except Exception as e:
		messagebox.showerror("Critical Error", str(e))
		root.destroy()


if __name__ == "__main__":
	main()