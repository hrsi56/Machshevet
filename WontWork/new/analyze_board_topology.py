import math

def analyze_board_topology(board):
    """
    Analyzes the board's topology to identify complex problematic states.
    Returns a dictionary of "badness" scores. Higher scores are worse.
    """
    peg_positions = board.all_pegs()
    peg_count = len(peg_positions)

    if peg_count < 2:
        return {"fragmentation_score": 0, "edge_penalty": 0, "pattern_penalty": 0}

    # --- 1. Fragmentation Score (Improved) ---
    # We check for connected components. Two pegs are "connected" if there's a
    # path between them consisting of steps of Manhattan distance <= 2.
    # This is a heuristic that approximates the ability to interact.
    q = [peg_positions[0]]
    visited = {peg_positions[0]}
    head = 0
    while head < len(q):
        r_curr, c_curr = q[head]
        head += 1
        for r_p, c_p in peg_positions:
            pos_p = (r_p, c_p)
            if pos_p in visited:
                continue
            # Manhattan distance <= 2 is a decent proxy for potential interaction
            if abs(r_p - r_curr) + abs(c_p - c_curr) <= 2:
                visited.add(pos_p)
                q.append(pos_p)

    fragmentation_score = 0
    if len(visited) < peg_count:
        # It's fragmented! The penalty is higher for smaller, isolated islands.
        # For simplicity, we'll start with a fixed high penalty for any fragmentation.
        fragmentation_score = 100 # High fixed score for being fragmented

    # --- 2. Edge Penalty: Quantifying Peg "Uselessness" ---
    # Pegs at the edge are less useful as tools to remove other pegs.
    edge_penalty = 0
    for r, c in peg_positions:
        # A peg's distance from the center (3,3) is a good measure of its utility
        distance_from_center = math.sqrt((r - 3)**2 + (c - 3)**2)
        # We use a non-linear penalty; being far from center is disproportionately worse.
        edge_penalty += distance_from_center ** 1.5

    # Normalize by number of pegs to get an average "badness" score
    edge_penalty = edge_penalty / peg_count

    # --- 3. Known "Pattern of Death" Penalty ---
    # The most famous anti-pattern is a 2x2 block of pegs, which is unsolvable.
    pattern_penalty = 0
    for r in range(6):
        for c in range(6):
            # Define the four positions of a potential 2x2 square
            square_positions = [(r, c), (r+1, c), (r, c+1), (r+1, c+1)]
            # Check how many of these positions are on the board and have pegs
            pegs_in_square = [pos for pos in square_positions if board.get(pos) == 1]
            if len(pegs_in_square) == 4:
                # A full 2x2 square is a catastrophe.
                pattern_penalty += 50 # High penalty

    return {
        "fragmentation_score": fragmentation_score,
        "edge_penalty": edge_penalty,
        "pattern_penalty": pattern_penalty
    }