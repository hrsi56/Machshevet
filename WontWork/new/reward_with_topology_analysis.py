from analyze_board_topology import analyze_board_topology


def reward_with_topology_analysis(board, move, done):
	"""
	A sophisticated reward function that uses the topology analysis engine.
	"""
	# --- End-of-Game states are absolute ---
	if done:
		if board.count_pegs() == 1 and board.get((3, 3)) == 1:
			return 1000.0
		else:
			return -200.0  # Heavy penalty for getting stuck

	# --- Analyze the health of the new board state ---
	topology = analyze_board_topology(board)

	# --- Apply penalties based on the analysis ---
	total_penalty = topology["fragmentation_score"] + topology["pattern_penalty"]

	# If we hit a catastrophic state, the penalty is overwhelming.
	if total_penalty > 0:
		return -total_penalty

	# --- If no catastrophic state, calculate the reward ---
	initial_pegs = 32
	moves_made = initial_pegs - board.count_pegs()

	# Exponential reward for progress
	base_reward = 1.2 ** moves_made

	# Subtract the more nuanced "edge_penalty" from the base reward.
	# This encourages keeping pegs centered and useful.
	final_reward = base_reward - topology["edge_penalty"]

	return final_reward