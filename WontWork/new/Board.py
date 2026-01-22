class Board:
    """
    Board class for Peg Solitaire.
    Manages the internal state of the board (positions of pegs and holes).
    Does NOT enforce move legality – only state management.
    """

    # קואורדינטות חוקיות בלוח 7x7, בצורה של צלב (33 חורים)
    LEGAL_POSITIONS = [
        (r, c) for r in range(7) for c in range(7)
        if not ((r < 2 or r > 4) and (c < 2 or c > 4))
    ]

    def __init__(self):
        """
        Initializes a standard Peg Solitaire board (cross pattern).
        All pegs are set except the center position.
        """
        self.state = {}
        for pos in self.LEGAL_POSITIONS:
            self.state[pos] = 1  # 1=peg, 0=hole

        self.state[(3, 3)] = 0  # Center is empty

    def get(self, pos):
        """
        Returns 1 if peg is present at position, 0 if hole, None if illegal position.
        """
        return self.state.get(pos, None)

    def set(self, pos, value):
        """
        Sets a position to 1 (peg) or 0 (hole).
        """
        if pos in self.LEGAL_POSITIONS and value in (0, 1):
            self.state[pos] = value
        else:
            raise ValueError(f"Illegal position or value: {pos}, {value}")

    def all_pegs(self):
        """
        Returns a list of all positions with pegs.
        """
        return [pos for pos, val in self.state.items() if val == 1]

    def all_holes(self):
        """
        Returns a list of all positions with holes.
        """
        return [pos for pos, val in self.state.items() if val == 0]

    def copy(self):
        """
        Returns a deep copy of the board.
        """
        new_board = Board()
        new_board.state = self.state.copy()
        return new_board

    def as_array(self):
        """
        Returns a 7x7 array with: 1=peg, 0=hole, -1=illegal position.
        Useful for neural net input.
        """
        arr = [[-1 for _ in range(7)] for _ in range(7)]
        for pos in self.LEGAL_POSITIONS:
            arr[pos[0]][pos[1]] = self.state[pos]
        return arr

    def count_pegs(self):
        """
        Returns the number of pegs currently on the board.
        """
        return sum(self.state[pos] for pos in self.LEGAL_POSITIONS)
    def to_dict(self):
        return self.state.copy()
    def __hash__(self):
        return hash(tuple(sorted(self.state.items())))

    def __eq__(self, other):
        return isinstance(other, Board) and self.state == other.state
    def set_state(self, state_dict):
        """Set board to a given dictionary of positions (for randomization/testing)."""
        for pos in self.LEGAL_POSITIONS:
            self.state[pos] = state_dict.get(pos, 0)