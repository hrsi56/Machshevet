import tkinter as tk
from tkinter import messagebox
import threading
import pickle
import os
import time
from collections import deque


class PegSolitaireSolver:
	MEMORY_FILE = "solitaire_raw_inline_brain.pkl"

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

		self.moves = []
		self.reverse_moves = []
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

						self.reverse_moves.append({
							'mask': mask,
							'req_peg': (1 << dst),
							'req_empty': (1 << src) | (1 << mid)
						})

		self.winning_states = set()
		self.loaded_from_disk = self.load_memory()

	def get_initial_board(self):
		return self.valid_mask & ~(1 << self.center_bit)

	def save_memory(self):
		print(f"💾 Saving {len(self.winning_states)} raw states...")
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
				print(f"✅ Loaded {len(self.winning_states)} states (Raw Mode).")
				return True
			except:
				return False
		return False

	def train(self):
		print("🧠 Starting Raw Training (Inlined - No Function Overhead)...")
		start_time = time.time()

		# Phase 1: Forward Reachability (Filter)
		print("   Phase 1: Mapping reachable universe...")
		start_board = self.get_initial_board()
		reachable_canonicals = {start_board}
		queue = deque([start_board])

		count = 0
		while queue:
			current_board = queue.popleft()
			count += 1
			if count % 50000 == 0:
				print(f"   Processed {count} states...")

			for m in self.moves:
				if (current_board & m['check_src'] == m['check_src']) and (current_board & m['check_dst'] == 0):
					next_board = current_board ^ m['mask']
					if next_board not in reachable_canonicals:
						reachable_canonicals.add(next_board)
						queue.append(next_board)

		print(f"   Phase 1 Complete. Total Reachable States: {len(reachable_canonicals)}")

		# Phase 2: Reverse Solving
		print("   Phase 2: Backtracking from win condition...")
		end_state = (1 << self.center_bit)
		if end_state not in reachable_canonicals:
			print("❌ Error: Winning state is theoretically impossible from start!")
			return

		self.winning_states = {end_state}
		queue = deque([end_state])

		processed = 0
		while queue:
			current_board = queue.popleft()
			processed += 1
			for m in self.reverse_moves:
				if (current_board & m['req_peg']) and (current_board & m['req_empty'] == 0):
					prev_board = current_board ^ m['mask']
					if prev_board not in self.winning_states:
						if prev_board in reachable_canonicals:
							self.winning_states.add(prev_board)
							queue.append(prev_board)

		duration = time.time() - start_time
		print(f"✅ Training Complete in {duration:.2f}s.")
		print(f"   Final Database Size: {len(self.winning_states)} unique raw states.")
		self.save_memory()

	def get_winning_moves_count(self, board):
		count = 0
		for m in self.moves:
			if (board & m['check_src'] == m['check_src']) and (board & m['check_dst'] == 0):
				next_board = board ^ m['mask']
				if next_board in self.winning_states:
					count += 1
		return count

	def solve_full_path(self, start_board):
		path = []
		if self._find_path_forward(start_board, path):
			return self._reconstruct(start_board, path)
		return None

	def _find_path_forward(self, board, path):
		if board == (1 << self.center_bit): return True
		if board not in self.winning_states: return False

		candidates = []
		for m in self.moves:
			if (board & m['check_src'] == m['check_src']) and (board & m['check_dst'] == 0):
				next_board = board ^ m['mask']
				if next_board in self.winning_states:
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

		msg = f"Winning Moves: {curr}" if curr > 0 else "DEAD END"
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
		self.root.title("Solitaire Pro (RAW & INLINED)")
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
		for r in range(7):
			for c in range(7):
				if (r, c) not in self.solver.r_c_to_bit: continue
				idx = self.solver.r_c_to_bit[(r, c)]
				has_peg = (self.current_board >> idx) & 1

				x = self.PADDING + c * self.CELL_SIZE + self.CELL_SIZE // 2
				y = self.PADDING + r * self.CELL_SIZE + self.CELL_SIZE // 2
				fill = self.COLOR_PEG if has_peg else self.COLOR_HOLE
				out, width = "", 1

				if has_peg and self.selected == (r, c):
					fill, out, width = self.COLOR_SELECTED, "white", 2
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
		self.graph.update_graph(wins)
		self.draw_board()
		self.root.after(200, lambda: self._animate(states, idx + 1))


def main():
	root = tk.Tk()
	root.withdraw()
	solver = PegSolitaireSolver()

	if not solver.loaded_from_disk:
		splash = tk.Toplevel(root)
		splash.geometry("400x180")
		splash.overrideredirect(True)
		splash.configure(bg="#222")

		tk.Label(splash, text="Training Brain...", font=("Segoe UI", 18, "bold"), fg="white", bg="#222").pack(
			pady=(40, 10))
		tk.Label(splash, text="Running RAW MODE (No Symmetry, No Calls)\nHigh RAM usage expected.",
		         font=("Segoe UI", 10), fg="#ff5555", bg="#222").pack()
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