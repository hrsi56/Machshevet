import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

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


class Game:
    """
    Game logic for Peg Solitaire.
    Holds a Board, manages move legality, applying moves, undo/redo, reward, and end-of-game conditions.
    """

    DIRECTIONS = [(-2, 0), (2, 0), (0, -2), (0, 2)]

    def __init__(self, board=None, reward_fn=None):
        """
        Initializes the game.
        board: Board object to start from (default: standard).
        reward_fn: Callable (state, move, done) -> float, for RL training.
        """
        self.board = board.copy() if board else Board()
        self.move_history = []         # [(from_pos, to_pos, over_pos, Board-state-before)]
        self.redo_stack = []           # for redo functionality
        self.last_move = None
        self.reward_fn = reward_fn     # RL: custom reward function
        self.custom_metadata = {}      # extra info (for RL/debug/stats)
        self.move_log = []             # for logging all moves (can be exported)

    def is_legal_move(self, from_pos, to_pos):
        if (from_pos not in Board.LEGAL_POSITIONS) or (to_pos not in Board.LEGAL_POSITIONS):
            return (False, None)
        dr = to_pos[0] - from_pos[0]
        dc = to_pos[1] - from_pos[1]
        if (abs(dr), abs(dc)) not in [(2, 0), (0, 2)]:
            return (False, None)
        over_pos = ((from_pos[0] + to_pos[0]) // 2, (from_pos[1] + to_pos[1]) // 2)
        if (self.board.get(from_pos) != 1 or
            self.board.get(over_pos) != 1 or
            self.board.get(to_pos) != 0):
            return (False, None)
        return (True, over_pos)

    def get_legal_moves(self):
        moves = []
        for from_pos in Board.LEGAL_POSITIONS:
            if self.board.get(from_pos) != 1:
                continue
            for d in self.DIRECTIONS:
                to_pos = (from_pos[0] + d[0], from_pos[1] + d[1])
                legal, over_pos = self.is_legal_move(from_pos, to_pos)
                if legal:
                    moves.append((from_pos, to_pos, over_pos))
        return moves

    def apply_move(self, from_pos, to_pos):
        """
        Applies a legal move (if valid), records state for undo/redo.
        Returns (applied:bool, reward:float, done:bool, info:dict).
        """
        legal, over_pos = self.is_legal_move(from_pos, to_pos)
        if not legal:
            return False, 0.0, self.is_game_over(), {"reason": "Illegal move"}
        # Store history for undo: deep copy of board state
        board_before = self.board.copy()
        self.move_history.append((from_pos, to_pos, over_pos, board_before))
        self.redo_stack.clear()  # clear redo stack after new move
        # Apply move
        self.board.set(from_pos, 0)
        self.board.set(over_pos, 0)
        self.board.set(to_pos, 1)
        self.last_move = (from_pos, over_pos, to_pos)
        self.move_log.append(self.last_move)
        # Compute reward
        reward = 0.0
        done = self.is_game_over()
        if self.reward_fn:
            reward = self.reward_fn(self.board.copy(), self.last_move, done)
        info = {"last_move": self.last_move, "done": done}
        return True, reward, done, info

    def undo(self):
        """
        Undo the last move, if possible. Returns True if undone.
        """
        if not self.move_history:
            return False
        (from_pos, to_pos, over_pos, board_before) = self.move_history.pop()
        self.redo_stack.append((from_pos, to_pos, over_pos, self.board.copy()))
        self.board = board_before
        self.last_move = self.move_history[-1][:3] if self.move_history else None
        return True

    def redo(self):
        """
        Redo a move that was undone, if possible. Returns True if redone.
        """
        if not self.redo_stack:
            return False
        (from_pos, to_pos, over_pos, board_before) = self.redo_stack.pop()
        self.move_history.append((from_pos, to_pos, over_pos, board_before))
        self.board = board_before.copy()
        self.apply_move(from_pos, to_pos)
        return True

    def is_game_over(self):
        return len(self.get_legal_moves()) == 0

    def is_win(self):
        return (self.board.count_pegs() == 1) and (self.board.get((3, 3)) == 1)

    def reset(self, board=None):
        self.board = board.copy() if board else Board()
        self.move_history.clear()
        self.redo_stack.clear()
        self.last_move = None
        self.move_log.clear()
        self.custom_metadata.clear()

    def get_state(self):
        return self.board.copy()

    def set_state(self, state):
        """
        Sets board to a specific Board object or dictionary state.
        """
        if isinstance(state, Board):
            self.board = state.copy()
        elif isinstance(state, dict):
            self.board.set_state(state)
        else:
            raise ValueError("Invalid state type for set_state")
        self.move_history.clear()
        self.redo_stack.clear()
        self.last_move = None

    def __hash__(self):
        return hash(self.board)

    def __eq__(self, other):
        return isinstance(other, Game) and self.board == other.board

    def export_move_log(self):
        """
        Returns the move log (list of moves) for analysis or training.
        """
        return list(self.move_log)

    def get_custom_metadata(self, key, default=None):
        return self.custom_metadata.get(key, default)

    def set_custom_metadata(self, key, value):
        self.custom_metadata[key] = value

from streamlit_drawable_canvas import st_canvas

CELL_SIZE = 60
GRID_SIZE = 7
CANVAS_SIZE = GRID_SIZE * CELL_SIZE
PEG_RADIUS = 20

# אתחול מצב
if "game" not in st.session_state:
    st.session_state.game = Game()
if "selected" not in st.session_state:
    st.session_state.selected = None

st.title("🧠 מחשבת – Peg Solitaire עם לחיצה על הלוח")

# ציור לוח כקנבס
canvas_result = st_canvas(
    fill_color="rgba(255, 214, 0, 1)",  # צבע פיון
    stroke_width=2,
    background_color="#eeeeee",
    update_streamlit=True,
    height=CANVAS_SIZE,
    width=CANVAS_SIZE,
    drawing_mode="transform",  # לא מציירים – רק לחיצה
    key="canvas",
)

# ציור גרפי
import matplotlib.pyplot as plt

def draw_board(canvas):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')
    ax.axis('off')

    for pos in Board.LEGAL_POSITIONS:
        r, c = pos
        val = st.session_state.game.board.get(pos)
        x = c + 0.5
        y = 6.5 - r  # כדי ש־(0,0) יהיה למעלה
        color = "#FFD600" if val == 1 else "#202020"
        edge = "#42A5F5" if st.session_state.selected == pos else "black"
        circ = plt.Circle((x, y), 0.3, color=color, ec=edge, lw=2)
        ax.add_patch(circ)

    st.pyplot(fig)

draw_board(canvas_result)

# לחיצה: מחשוב תא
if canvas_result and canvas_result.json_data and "objects" in canvas_result.json_data:
    # קבלת קואורדינטות לחיצה אחרונה
    objs = canvas_result.json_data["objects"]
    if objs:
        obj = objs[-1]
        x, y = obj["left"], obj["top"]
        col = int(x // CELL_SIZE)
        row = int(y // CELL_SIZE)
        clicked = (row, col)

        if clicked in Board.LEGAL_POSITIONS:
            if st.session_state.selected is None:
                # שלב ראשון – בחירת פיון
                if st.session_state.game.board.get(clicked) == 1:
                    st.session_state.selected = clicked
                    st.info(f"נבחר פיון מ: {clicked}")
            else:
                # שלב שני – ניסיון קפיצה
                from_pos = st.session_state.selected
                to_pos = clicked
                applied, reward, done, info = st.session_state.game.apply_move(from_pos, to_pos)
                if applied:
                    st.success(f"המהלך בוצע: {from_pos} ➝ {to_pos}")
                else:
                    st.error("מהלך לא חוקי!")
                st.session_state.selected = None
                st.rerun()

# כפתורי שליטה
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("↩️ Undo"):
        st.session_state.game.undo()
        st.session_state.selected = None
        st.rerun()
with col2:
    if st.button("↪️ Redo"):
        st.session_state.game.redo()
        st.session_state.selected = None
        st.rerun()
with col3:
    if st.button("🔄 Reset"):
        st.session_state.game.reset()
        st.session_state.selected = None
        st.rerun()

# סטטוס
peg_count = st.session_state.game.board.count_pegs()
move_count = len(st.session_state.game.move_log)
if st.session_state.game.is_win():
    st.success("🎉 ניצחון! פיון יחיד במרכז!")
elif st.session_state.game.is_game_over():
    st.warning("🛑 אין מהלכים חוקיים – נסה מחדש.")
else:
    st.info(f"פינים: {peg_count} | מהלכים: {move_count}")


# לוג מהלכים
with st.expander("📜 לוג מהלכים"):
    for idx, (f, o, t) in enumerate(st.session_state.game.export_move_log(), 1):
        st.markdown(f"{idx:2}: {f} ➝ {t} (דרך {o})")