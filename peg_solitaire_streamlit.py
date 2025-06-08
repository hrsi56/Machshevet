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
from PIL import Image, ImageDraw

# ---------- קבועים גרפיים ----------
GRID = 7
CELL = 70                     # פיקסלים לתא אחד
RADIUS = int(CELL*0.38)       # חישוב רדיוס מעגל
SIZE  = GRID * CELL           # גודל הקנבס (ריבוע)

COL_PEG   = "#FFD600"
COL_HOLE  = "#202020"
COL_EDGE  = "black"
COL_SEL   = "#42A5F5"

# ---------- אתחול מצב ----------
if "game" not in st.session_state:
    st.session_state.game = Game()
if "sel" not in st.session_state:     # פיון שנבחר
    st.session_state.sel = None

# ---------- פונקציה שמחזירה תמונה עדכנית של הלוח ----------
def render_board(board: Board, selected):
    img = Image.new("RGB", (SIZE, SIZE), "#eeeeee")
    draw = ImageDraw.Draw(img)

    for (r, c) in Board.LEGAL_POSITIONS:
        x = c*CELL + CELL//2
        y = r*CELL + CELL//2
        fill = COL_PEG if board.get((r, c)) == 1 else COL_HOLE
        outline = COL_SEL if selected==(r, c) else COL_EDGE
        draw.ellipse(
            (x-RADIUS, y-RADIUS, x+RADIUS, y+RADIUS),
            fill=fill, outline=outline, width=3
        )
    return img

# ---------- כותרת ----------
st.title("🧠 מחשבת – Peg Solitaire בלחיצה על הלוח")

# ---------- ציור הלוח כרקע של Canvas ----------
board_img = render_board(st.session_state.game.board, st.session_state.sel)
canvas = st_canvas(
    background_image=board_img,
    update_streamlit=True,
    height=SIZE,
    width=SIZE,
    drawing_mode="transform",   # לא מוסיף צורות, רק קליקים
    key="board_canvas"
)

# ---------- טיפול בלחיצה ----------
if canvas.json_data and canvas.json_data["objects"]:
    obj = canvas.json_data["objects"][-1]
    # קואורדינטות יחסיות
    x, y = obj["left"], obj["top"]
    col = int(x // CELL)
    row = int(y // CELL)
    pos = (row, col)

    if pos in Board.LEGAL_POSITIONS:
        game = st.session_state.game
        if st.session_state.sel is None:                # לחיצה ראשונה → בחר פין
            if game.board.get(pos) == 1:
                st.session_state.sel = pos
                st.experimental_rerun()                 # צייר מחדש עם קו מתאר כחול
        else:                                           # לחיצה שנייה → נסה מהלך
            from_pos = st.session_state.sel
            to_pos   = pos
            applied, _, _, _ = game.apply_move(from_pos, to_pos)
            st.session_state.sel = None
            if applied:
                st.success(f"המהלך בוצע: {from_pos} ➝ {to_pos}")
            else:
                st.error("מהלך לא חוקי")
            st.experimental_rerun()                     # רענן תצוגה

# ---------- כפתורי שליטה ----------
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("↩️ Undo", disabled=not st.session_state.game.move_history):
        st.session_state.game.undo()
        st.experimental_rerun()
with col2:
    if st.button("↪️ Redo", disabled=not st.session_state.game.redo_stack):
        st.session_state.game.redo()
        st.experimental_rerun()
with col3:
    if st.button("🔄 Reset"):
        st.session_state.game.reset()
        st.experimental_rerun()

# ---------- סטטוס ולוג ----------
game = st.session_state.game
peg_cnt  = game.board.count_pegs()
move_cnt = len(game.move_log)

if game.is_win():
    st.success("🎉 ניצחון! פיון יחיד במרכז!")
elif game.is_game_over():
    st.warning("🛑 אין מהלכים חוקיים – נסה מחדש.")
else:
    st.info(f"פינים: {peg_cnt} | מהלכים: {move_cnt}")

with st.expander("📜 לוג מהלכים"):
    for i, (f, o, t) in enumerate(game.export_move_log(), 1):
        st.write(f"{i:2}: {f} ➝ {t} (דרך {o})")


