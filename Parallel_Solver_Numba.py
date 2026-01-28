import tkinter as tk
from tkinter import messagebox
import threading
import pickle
import os
import time
from collections import deque
import numpy as np
from numba import njit, int64, prange, get_num_threads


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


# Accepts 'out_buffer' externally to avoid repeated memory allocation overhead in the hot loop.
@njit(parallel=True, nogil=True)
def expand_batch_parallel(boards, table, moves_mask, checks_src, checks_dst, out_buffer):
	num_boards = len(boards)
	num_moves = len(moves_mask)

	# Direct write into pre-allocated memory block to prevent GC pressure
	for i in prange(num_boards):
		curr = boards[i]
		for m in range(num_moves):
			idx = i * num_moves + m
			if (curr & checks_src[m] == checks_src[m]) and (curr & checks_dst[m] == 0):
				next_board = curr ^ moves_mask[m]
				out_buffer[idx] = fast_canonical_lookup_single(next_board, table)


class PegSolitaireSolver:
	MEMORY_FILE = "solitaire_parallel_brain.pkl"
	BATCH_SIZE = 25000

	def __init__(self):
		self.r_c_to_bit = {}
		self.bit_to_r_c = {}
		self.valid_mask = 0
		self.center_bit = 0

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

		self.moves_mask = []
		self.moves_check_src = []
		self.moves_check_dst = []
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
						self.moves_mask.append(mask)
						self.moves_check_src.append((1 << src) | (1 << mid))
						self.moves_check_dst.append(1 << dst)
						self.moves_metadata.append({
							'mask': mask, 'check_src': (1 << src) | (1 << mid),
							'check_dst': (1 << dst), 'src': src, 'dst': dst
						})

		self.moves_mask = np.array(self.moves_mask, dtype=np.int64)
		self.moves_check_src = np.array(self.moves_check_src, dtype=np.int64)
		self.moves_check_dst = np.array(self.moves_check_dst, dtype=np.int64)

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
					mapping[self.r_c_to_bit[(r, c)]] = self.r_c_to_bit[(rr, cc)]
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
						real_idx = (lsb.bit_length() - 1) + bit_offset
						if real_idx in mapping:
							transformed_val |= (1 << mapping[real_idx])
						t ^= lsb
					table[sym_idx, chunk_id, val] = transformed_val
		return table

	def get_canonical(self, board):
		return fast_canonical_lookup_single(board, self.lookup_table)

	def get_initial_board(self):
		return self.valid_mask & ~(1 << self.center_bit)

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

		# PHASE 1: Forward BFS with Parallel Batching
		start_board = self.get_initial_board()
		canon_start = self.get_canonical(start_board)
		reachable_canonicals = {canon_start}
		queue = deque([start_board])
		count = 0

		# Optimization: One-time Memory Allocation.
		# This shared buffer is recycled in every iteration to avoid malloc overhead.
		num_possible_moves = len(self.moves_mask)
		buffer_size = self.BATCH_SIZE * num_possible_moves
		shared_buffer = np.zeros(buffer_size, dtype=np.int64)
		print(f"   Allocated shared buffer: {buffer_size * 8 / 1024 / 1024:.2f} MB")

		while queue:
			batch = []
			for _ in range(min(len(queue), self.BATCH_SIZE)):
				batch.append(queue.popleft())
			if not batch: break

			batch_np = np.array(batch, dtype=np.int64)
			current_batch_size = len(batch_np)

			expand_batch_parallel(
				batch_np,
				self.lookup_table,
				self.moves_mask,
				self.moves_check_src,
				self.moves_check_dst,
				shared_buffer
			)

			# Slicing the valid part of the buffer.
			# We only read what was potentially written in this batch.
			relevant_size = current_batch_size * num_possible_moves
			valid_results_view = shared_buffer[:relevant_size]
			unique_candidates = np.unique(valid_results_view)

			# Clean buffer for next iteration
			valid_results_view.fill(0)

			for canon_next in unique_candidates:
				if canon_next == 0: continue
				if canon_next not in reachable_canonicals:
					reachable_canonicals.add(canon_next)
					queue.append(canon_next)

			count += len(batch)
			if count % 100000 < self.BATCH_SIZE:
				print(f"   Processed {count} states... Queue: {len(queue)}")

		# PHASE 2: Reverse Solving (Backtracking from Win)
		print("   Phase 2: Backtracking...")
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

		print(f"✅ Training Complete in {time.time() - start_time:.2f}s.")
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

		curr_data = self.data_points[-1]
		curr_val = curr_data['val']
		is_win = curr_data['win']

		color = "#00E676"
		msg = f"Winning Moves: {curr_val}"

		if is_win:
			msg, color = "VICTORY!", "#FFD700"
		elif curr_val == 0:
			msg, color = "DEAD END", "#D50000"
		elif curr_val == 1:
			color = "#FF9100"

		step = self.w / max(32, len(self.data_points))
		coords = []

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
		self.graph.update_graph(self.solver.get_winning_moves_count(self.current_board))

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
		for r, c in self.solver.r_c_to_bit:
			idx = self.solver.r_c_to_bit[(r, c)]
			has_peg = (self.current_board >> idx) & 1
			x = self.PADDING + c * self.CELL_SIZE + self.CELL_SIZE // 2
			y = self.PADDING + r * self.CELL_SIZE + self.CELL_SIZE // 2
			fill = self.COLOR_SELECTED if has_peg and self.selected == (r, c) else (
				self.COLOR_PEG if has_peg else self.COLOR_HOLE)
			width = 2 if fill == self.COLOR_SELECTED else 1
			self.cvs.create_oval(x - 18, y - 18, x + 18, y + 18, fill=fill, outline="" if width == 1 else "white",
			                     width=width)

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
		move = next((m for m in self.solver.moves_metadata if m['src'] == s_idx and m['dst'] == d_idx), None)

		if move and (self.current_board & (move['check_src'] ^ (1 << s_idx))):
			self.history.append(self.current_board)
			self.current_board ^= move['mask']
			self.selected = None

			is_victory = (self.current_board == (1 << self.solver.center_bit))
			wins = self.solver.get_winning_moves_count(self.current_board)

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
		self.graph.update_graph(self.solver.get_winning_moves_count(self.current_board))
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
		self.graph.update_graph(wins, is_win=(self.current_board == (1 << self.solver.center_bit)))
		self.draw_board()
		self.root.after(200, lambda: self._animate(states, idx + 1))


def main():
	root = tk.Tk()
	root.withdraw()
	solver = PegSolitaireSolver()

	if not solver.loaded_from_disk:
		splash = tk.Toplevel(root)
		splash.geometry("480x200")
		splash.overrideredirect(True)
		splash.configure(bg="#222")

		tk.Label(splash, text="Training AI Brain...", font=("Segoe UI", 18, "bold"), fg="white", bg="#222").pack(
			pady=(40, 10))
		tk.Label(splash,
		         text=f"PARALLEL MODE: Using {get_num_threads()} CPU Cores via Numba\nPhase 1: Parallel Batching | Phase 2: Reverse",
		         font=("Segoe UI", 10), fg="#00E676", bg="#222").pack()
		root.update()

		def run_train():
			solver.train()
			root.after(0, lambda: [splash.destroy(), root.deiconify(), root.geometry("650x550"),
			                       PegSolitaireGUI(root, solver)])

		threading.Thread(target=run_train).start()
	else:
		root.deiconify()
		root.geometry("650x550")
		PegSolitaireGUI(root, solver)

	root.mainloop()


if __name__ == "__main__":
	main()