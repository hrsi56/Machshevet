import pickle
import numpy as np
import matplotlib.pyplot as plt
import os


class BrainAnalytics:
	def __init__(self, memory_file="solitaire_parallel_brain.pkl"):
		print(f"📂 Loading brain from {memory_file}...")
		if not os.path.exists(memory_file):
			print("❌ Error: Brain file not found. Run main.py first.")
			exit()

		with open(memory_file, "rb") as f:
			self.winning_states = pickle.load(f)

		self.states_array = np.array(list(self.winning_states), dtype=np.int64)
		print(f"✅ Loaded {len(self.states_array):,} winning states.")

		self.r_c_to_bit = {}
		self._init_mapping()

		# Pre-calculate turns (32 = Start, 1 = End)
		print("🧮 Calculating turn data...")
		self.pop_counts = np.zeros(len(self.states_array), dtype=int)
		temp = self.states_array.copy()
		while np.any(temp):
			self.pop_counts += (temp & 1)
			temp >>= 1
		self.turns = 32 - self.pop_counts

	def _init_mapping(self):
		idx = 0
		for r in range(7):
			for c in range(7):
				if not ((r < 2 and c < 2) or (r < 2 and c > 4) or
				        (r > 4 and c < 2) or (r > 4 and c > 4)):
					self.r_c_to_bit[(r, c)] = idx
					idx += 1

	def save_plot(self, filename):
		print(f"💾 Saving {filename}...")
		plt.savefig(filename, dpi=100, bbox_inches='tight')
		plt.close()

	def generate_histogram(self):
		print("📊 Generating Distribution Histogram...")
		unique, counts = np.unique(self.pop_counts, return_counts=True)
		dist = dict(zip(unique, counts))

		plt.figure(figsize=(10, 6))
		bars = plt.bar(dist.keys(), dist.values(), color='#00E676', edgecolor='black')

		plt.title('The "Belly" of the Game: Winning States per Peg Count', fontsize=16, fontweight='bold')
		plt.xlabel('Number of Pegs on Board', fontsize=12)
		plt.ylabel('Unique Winning States', fontsize=12)
		plt.grid(axis='y', alpha=0.3)
		plt.xticks(range(1, 33, 2))

		for bar in bars:
			height = bar.get_height()
			if height > 5000:
				plt.text(bar.get_x() + bar.get_width() / 2., height,
				         f'{int(height / 1000)}k', ha='center', va='bottom', fontsize=8)

		self.save_plot("analytics_distribution.png")

	def generate_timeline_maps(self):
		print("⏳ Generating Awakening & Last Stand Maps...")

		first_change_map = np.full((7, 7), 32.0)
		last_active_map = np.zeros((7, 7))

		# Start mask (Full board except center)
		start_mask = np.zeros((7, 7), dtype=bool)
		for r in range(7):
			for c in range(7):
				if (r, c) in self.r_c_to_bit: start_mask[r, c] = True
		start_mask[3, 3] = False

		for r in range(7):
			for c in range(7):
				if (r, c) not in self.r_c_to_bit: continue
				bit_idx = self.r_c_to_bit[(r, c)]
				mask = 1 << bit_idx
				has_peg = (self.states_array & mask) > 0

				# 1. The Awakening (Min Turn of change)
				changed = np.where(has_peg != start_mask[r, c])[0]
				if len(changed) > 0:
					first_change_map[r, c] = np.min(self.turns[changed])
				else:
					first_change_map[r, c] = 0

				# 2. The Last Stand (Max Turn active)
				active = np.where(has_peg)[0]
				if len(active) > 0:
					last_active_map[r, c] = np.max(self.turns[active])

		# Plotting
		fig, axes = plt.subplots(1, 2, figsize=(16, 7))

		# Map 1
		im1 = axes[0].imshow(first_change_map, cmap='coolwarm', vmin=0, vmax=20)
		axes[0].set_title("The Awakening: Earliest Turn of Change\n(Blue = Immediate Action)", fontsize=12,
		                  fontweight='bold')
		axes[0].axis('off')

		# Map 2
		im2 = axes[1].imshow(last_active_map, cmap='turbo', vmin=10, vmax=32)
		axes[1].set_title("The Last Stand: Latest Turn with a Peg\n(Red = The Last Survivor)", fontsize=12,
		                  fontweight='bold')
		axes[1].axis('off')

		for ax, data in zip(axes, [first_change_map, last_active_map]):
			for r in range(7):
				for c in range(7):
					if (r, c) in self.r_c_to_bit:
						val = data[r, c]
						color = "white" if (val < 10 or val > 25) else "black"
						ax.text(c, r, f"T{int(val)}", ha="center", va="center", color=color, fontsize=9,
						        fontweight='bold')

		self.save_plot("analytics_timeline.png")

	def generate_expected_flux(self):
		print("🔥 Generating Time Center of Gravity...")

		prob_timeline = np.zeros((33, 7, 7))
		for t in range(33):
			indices = np.where(self.turns == t)[0]
			if len(indices) == 0: continue
			subset = self.states_array[indices]
			for r in range(7):
				for c in range(7):
					if (r, c) in self.r_c_to_bit:
						mask = 1 << self.r_c_to_bit[(r, c)]
						prob_timeline[t, r, c] = np.count_nonzero(subset & mask) / len(subset)

		# Flux = Absolute difference between turns
		flux_timeline = np.abs(np.diff(prob_timeline, axis=0))

		# Weighted Average
		turn_indices = np.arange(1, 33).reshape(-1, 1, 1)
		numerator = np.sum(flux_timeline * turn_indices, axis=0)
		denominator = np.sum(flux_timeline, axis=0)

		with np.errstate(divide='ignore', invalid='ignore'):
			expected_map = numerator / denominator
			expected_map[np.isnan(expected_map)] = 0

		plt.figure(figsize=(8, 8))
		plt.imshow(expected_map, cmap='plasma', vmin=0, vmax=32)
		plt.title('Time Center of Gravity: When does the action happen?\n(Blue = Early Game, Yellow = End Game)',
		          fontsize=14, fontweight='bold')
		plt.axis('off')

		for r in range(7):
			for c in range(7):
				if (r, c) in self.r_c_to_bit:
					val = expected_map[r, c]
					color = "black" if (10 < val < 25) else "white"
					plt.text(c, r, f"T{val:.1f}", ha="center", va="center", color=color, fontsize=9, fontweight='bold')

		self.save_plot("analytics_flux_time.png")


if __name__ == "__main__":
	analytics = BrainAnalytics()
	analytics.generate_histogram()
	analytics.generate_timeline_maps()
	analytics.generate_expected_flux()
	print("🚀 All analytics generated successfully!")