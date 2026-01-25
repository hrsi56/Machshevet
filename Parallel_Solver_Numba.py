import tkinter as tk
from tkinter import messagebox
import threading
import pickle
import os
import time
from collections import deque
import numpy as np
from numba import njit, int64, prange, get_num_threads


# ==========================================
#  חלק 0: המנוע הגרעיני (Numba Parallel Kernel)
# ==========================================

# 1. פונקציית עזר לחישוב קנוני (רצה בתוך כל ליבה)
@njit(int64(int64, int64[:, :, :]), nogil=True)
def fast_canonical_lookup_single(board, table):
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


# 2. הלב המקבילי: מעבד Batch שלם של לוחות על כל הליבות
@njit(parallel=True, nogil=True)
def expand_batch_parallel(boards, table, moves_mask, checks_src, checks_dst):
	num_boards = len(boards)
	num_moves = len(moves_mask)

	# הקצאת מערך תוצאות שטוח (כל לוח * מספר המהלכים האפשריים)
	max_results = num_boards * num_moves
	results = np.zeros(max_results, dtype=np.int64)

	# prange = Parallel Range. נומבה מחלקת את האיטרציות בין הליבות
	for i in prange(num_boards):
		curr = boards[i]

		for m in range(num_moves):
			# בדיקת חוקיות המהלך (Bitwise)
			if (curr & checks_src[m] == checks_src[m]) and (curr & checks_dst[m] == 0):
				next_board = curr ^ moves_mask[m]

				# חישוב קנוני מיידי
				canon = fast_canonical_lookup_single(next_board, table)

				# שמירה במיקום ייחודי במערך התוצאות (למניעת התנגשויות)
				idx = i * num_moves + m
				results[idx] = canon
			else:
				# סימון 0 למצב לא חוקי
				results[i * num_moves + m] = 0

	return results


# ==========================================
#  חלק 1: המנהל (Solver Class)
# ==========================================
class PegSolitaireSolver:
	MEMORY_FILE = "solitaire_parallel_brain.pkl"
	BATCH_SIZE = 25000  # גודל הנגלה (Batch) לעיבוד מקבילי

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

		# המרת מהלכים למערכי Numpy (עבור Numba)
		self.moves_mask = []
		self.moves_check_src = []
		self.moves_check_dst = []

		# שמירת מהלכים כמילונים (עבור ה-GUI והשחזור)
		self.moves_metadata = []

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

						# Numpy Lists
						self.moves_mask.append(mask)
						self.moves_check_src.append((1 << src) | (1 << mid))
						self.moves_check_dst.append(1 << dst)

						# Metadata List
						self.moves_metadata.append({
							'mask': mask, 'check_src': (1 << src) | (1 << mid),
							'check_dst': (1 << dst), 'src': src, 'dst': dst
						})

		# Finalize Numpy Arrays
		self.moves_mask = np.array(self.moves_mask, dtype=np.int64)
		self.moves_check_src = np.array(self.moves_check_src, dtype=np.int64)
		self.moves_check_dst = np.array(self.moves_check_dst, dtype=np.int64)

		# Reverse Moves (For Phase 2)
		self.reverse_moves = []
		for i in range(len(self.moves_mask)):
			self.reverse_moves.append({
				'mask': self.moves_mask[i],
				'req_peg': self.moves_check_dst[i],
				'req_empty': self.moves_check_src[i]
			})

		self.winning_states = set()
		self.loaded_from_disk = self.load_memory()

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
		print("⚡ Building Optimized Lookup Tables...")
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
		return fast_canonical_lookup_single(board, self.lookup_table)

	def get_initial_board(self):
		board = self.valid_mask
		board &= ~(1 << self.center_bit)
		return board

	def save_memory(self):
		print(f"💾 Saving {len(self.winning_states)} states...")
		try:
			with open(self.MEMORY_FILE, "wb") as f:
				pickle.dump(self.winning_states, f)
		except Exception as e:
			print(f"Save failed: {e}")

	def load_memory(self):
		if os.path.exists(self.MEMORY_FILE):
			try:
				with open(self.MEMORY_FILE, "rb") as f:
					self.winning_states = pickle.load(f)
				print(f"✅ Loaded {len(self.winning_states)} states.")
				return True
			except:
				return False
		return False

	def train(self):
		print(f"🧠 Starting PARALLEL Training on {get_num_threads()} CPU threads...")
		start_time = time.time()

		# ==========================================
		# PHASE 1: Parallel Forward BFS
		# ==========================================
		print("   Phase 1: Mapping reachable universe (Parallel Batching)...")

		start_board = self.get_initial_board()
		canon_start = self.get_canonical(start_board)

		reachable_canonicals = {canon_start}
		queue = deque([start_board])

		count = 0

		while queue:
			# 1. Collect Batch
			batch = []
			# משיכת כמות גדולה של לוחות מהתור
			for _ in range(min(len(queue), self.BATCH_SIZE)):
				batch.append(queue.popleft())

			if not batch: break

			# המרה למערך NumPy (כדי לשלוח ל-Numba)
			batch_np = np.array(batch, dtype=np.int64)

			# 2. Parallel Processing (The heavy lifting)
			# כאן קורה הקסם: כל הליבות עובדות במקביל
			results = expand_batch_parallel(batch_np, self.lookup_table,
			                                self.moves_mask, self.moves_check_src, self.moves_check_dst)

			# 3. Filter & Update (Back to Python main thread)
			# np.unique מסנן כפילויות הרבה יותר מהר מפייתון
			unique_candidates = np.unique(results)

			for canon_next in unique_candidates:
				if canon_next == 0: continue  # התעלמות מכישלונות

				if canon_next not in reachable_canonicals:
					reachable_canonicals.add(canon_next)
					# אנחנו יכולים להכניס את הקנוני לתור להמשך פיתוח
					queue.append(canon_next)

			count += len(batch)
			if count % 100000 < self.BATCH_SIZE:
				print(f"   Processed {count} states... Queue: {len(queue)}")

		print(f"   Phase 1 Complete. Reachable: {len(reachable_canonicals)}")

		# ==========================================
		# PHASE 2: Reverse Solving (Standard)
		# ==========================================
		print("   Phase 2: Backtracking from win condition...")

		end_state = (1 << self.center_bit)
		canon_end = self.get_canonical(end_state)

		if canon_end not in reachable_canonicals:
			print("Error: Impossible to win.")
			return

		self.winning_states = {canon_end}
		queue = deque([end_state])

		while queue:
			current_board = queue.popleft()

			for m in self.reverse_moves:
				if (current_board & m['req_peg']) and (current_board & m['req_empty'] == 0):
					prev_board = current_board ^ m['mask']
					canon_prev = self.get_canonical(prev_board)

					if canon_prev not in self.winning_states:
						if canon_prev in reachable_canonicals:
							self.winning_states.add(canon_prev)
							queue.append(prev_board)

		duration = time.time() - start_time
		print(f"✅ Training Complete in {duration:.2f}s.")
		self.save_memory()

	def get_winning_moves_count(self, board):
		count = 0
		for m in self.moves_metadata:
			if (board & m['check_src'] == m['check_src']) and (board & m['check_dst'] == 0):
				next_board = board ^ m['mask']
				if self.get_canonical(next_board) in self.winning_states:
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
		for m in self.moves_metadata:
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
#  GUI Components (Updated)
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

	# UPDATE: מקבל כעת פרמטר is_win כדי לדעת לצייר ניצחון
	def update_graph(self, val, is_win=False):
		self.data_points.append({'val': val, 'win': is_win})
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

		# שליפת המידע מהנקודה האחרונה
		curr_data = self.data_points[-1]
		curr_val = curr_data['val']
		is_win = curr_data['win']

		# קביעת הצבע והטקסט
		color = "#00E676"
		msg = f"Winning Moves: {curr_val}"

		if is_win:
			msg = "VICTORY!"
			color = "#FFD700"  # Gold
		elif curr_val == 0:
			msg = "DEAD END"
			color = "#D50000"
		elif curr_val == 1:
			color = "#FF9100"

		step = self.w / max(32, len(self.data_points))
		coords = []

		# ציור הגרף (צריך לחלץ את הערך המספרי מכל נקודה)
		for i, item in enumerate(self.data_points):
			val = item['val']
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

		self.create_text(10, 15, text=msg, anchor="w", fill="white", font=("Consolas", 10, "bold"), tag="text")


class PegSolitaireGUI:
	CELL_SIZE = 45
	PADDING = 10
	COLOR_BG = "#1e1e1e"
	COLOR_HOLE = "#333333"
	COLOR_PEG = "#FFD700"
	COLOR_SELECTED = "#2962FF"

	def __init__(self, root, solver):
		self.root = root
		self.solver = solver
		self.root.title("Solitaire Pro 🚀 (Parallel + Numba)")
		self.root.configure(bg=self.COLOR_BG)
		self.current_board = self.solver.get_initial_board()
		self.selected = None
		self.history = []
		self.animating = False
		self._init_ui()
		self.draw_board()

		initial_moves = self.solver.get_winning_moves_count(self.current_board)
		self.graph.update_graph(initial_moves, is_win=False)

	def _init_ui(self):
		main = tk.Frame(self.root, bg=self.COLOR_BG)
		main.pack(fill="both", expand=True, padx=20, pady=20)

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

		right = tk.Frame(main, bg=self.COLOR_BG)
		right.pack(side="right", fill="y", padx=(20, 0))
		tk.Label(right, text="Survival Funnel", font=("Segoe UI", 12, "bold"), fg="#aaa", bg=self.COLOR_BG).pack(
			anchor="w")
		self.graph = SurvivalFunnelGraph(right, width=250, height=150)
		self.graph.pack(pady=(5, 20))
		self.lbl_status = tk.Label(right, text="Game Start", font=("Segoe UI", 16, "bold"), fg="white",
		                           bg=self.COLOR_BG)
		self.lbl_status.pack(pady=10)

	def draw_board(self):
		self.cvs.delete("all")
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
				self.selected = (r, c)
			else:
				self.try_move(self.selected, (r, c))
		elif has_peg:
			self.selected = (r, c)
		self.draw_board()

	def try_move(self, src, dst):
		s_idx = self.solver.r_c_to_bit[src]
		d_idx = self.solver.r_c_to_bit[dst]
		move = None
		# חיפוש במטא-דאטה (המהלך האמיתי עם המסיכות והכל)
		for m in self.solver.moves_metadata:
			if m['src'] == s_idx and m['dst'] == d_idx:
				if (self.current_board & (m['check_src'] ^ (1 << s_idx))): move = m
				break
		if move:
			self.history.append(self.current_board)
			self.current_board ^= move['mask']
			self.selected = None

			# בדיקת ניצחון
			is_victory = (self.current_board == (1 << self.solver.center_bit))
			wins = self.solver.get_winning_moves_count(self.current_board)

			# עדכון הגרף עם סטטוס ניצחון
			self.graph.update_graph(wins, is_win=is_victory)

			self.draw_board()

			if is_victory:
				self.lbl_status.config(text="VICTORY! 🏆", fg="gold")
			elif wins > 0:
				self.lbl_status.config(text="Safe Move ✅", fg="#00E676")
			else:
				self.lbl_status.config(text="Dead End ❌", fg="#D50000")
		else:
			messagebox.showwarning("Oops", "Invalid move")

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

		# בדיקת ניצחון עבור האנימציה
		is_victory = (self.current_board == (1 << self.solver.center_bit))
		wins = self.solver.get_winning_moves_count(states[idx])

		self.graph.update_graph(wins, is_win=is_victory)
		self.draw_board()
		self.root.after(200, lambda: self._animate(states, idx + 1))


# ==========================================
#  Main Application
# ==========================================
def main():
	root = tk.Tk()
	root.withdraw()
	solver = PegSolitaireSolver()

	if not solver.loaded_from_disk:
		splash = tk.Toplevel(root)
		splash.geometry("480x200")
		splash.overrideredirect(True)
		splash.configure(bg="#222")

		lbl = tk.Label(splash, text="Training AI Brain...", font=("Segoe UI", 18, "bold"), fg="white", bg="#222")
		lbl.pack(pady=(40, 10))

		from numba import get_num_threads
		threads = get_num_threads()

		lbl_sub = tk.Label(splash,
		                   text=f"PARALLEL MODE: Using {threads} CPU Cores via Numba\nPhase 1: Parallel Batching | Phase 2: Reverse",
		                   font=("Segoe UI", 10), fg="#00E676", bg="#222")
		lbl_sub.pack()

		root.update()

		def run_train():
			solver.train()
			root.after(0, lambda: finish(root, splash, solver))

		threading.Thread(target=run_train).start()
	else:
		finish(root, None, solver)

	root.mainloop()


def finish(root, splash, solver):
	if splash: splash.destroy()
	root.deiconify()
	root.geometry("650x550")
	PegSolitaireGUI(root, solver)


if __name__ == "__main__":
	main()