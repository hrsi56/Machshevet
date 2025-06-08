import streamlit as st

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


import streamlit as st
from PIL import Image, ImageDraw


# הנח כאן את הגדרות המחלקות שלך: Board ו-Game
# לדוגמה (אני מניח מבנה בסיסי, התאם אותו לקוד שלך):

class Board:
    LEGAL_POSITIONS = {(r, c) for r in range(7) for c in range(7)} - \
                      {(r, c) for r in range(2) for c in range(2)} - \
                      {(r, c) for r in range(2) for c in range(5, 7)} - \
                      {(r, c) for r in range(5, 7) for c in range(2)} - \
                      {(r, c) for r in range(5, 7) for c in range(5, 7)}

    def __init__(self):
        self.grid = {pos: 1 for pos in self.LEGAL_POSITIONS}
        center = (3, 3)
        if center in self.grid:
            self.grid[center] = 0  # חור במרכז
        self._initial_state = self.grid.copy()

    def get(self, pos):
        return self.grid.get(pos)

    def set(self, pos, value):
        if pos in self.LEGAL_POSITIONS:
            self.grid[pos] = value

    def count_pegs(self):
        return sum(self.grid.values())

    def reset(self):
        self.grid = self._initial_state.copy()


class Game:
    def __init__(self):
        self.board = Board()
        self.move_history = []
        self.redo_stack = []
        self.move_log = []

    def apply_move(self, from_pos, to_pos):
        # ודא שהמיקומים חוקיים
        if from_pos not in Board.LEGAL_POSITIONS or to_pos not in Board.LEGAL_POSITIONS:
            return False, None, None, "מהלך לא חוקי: מחוץ ללוח"

        # בדוק אם יש פיון בנקודת ההתחלה ואין בנקודת הסיום
        if self.board.get(from_pos) != 1 or self.board.get(to_pos) != 0:
            return False, None, None, "מהלך לא חוקי: פיון התחלה או סיום לא תקינים"

        # חשב את מיקום הפיון ש"נאכל"
        dr, dc = to_pos[0] - from_pos[0], to_pos[1] - from_pos[1]
        if abs(dr) == 2 and dc == 0:
            over_pos = (from_pos[0] + dr // 2, from_pos[1])
        elif abs(dc) == 2 and dr == 0:
            over_pos = (from_pos[0], from_pos[1] + dc // 2)
        else:
            return False, None, None, "מהלך לא חוקי: תנועה לא באלכסון או רחוקה מדי"

        # ודא שקיים פיון שניתן לדלג מעליו
        if self.board.get(over_pos) != 1:
            return False, None, None, "מהלך לא חוקי: אין פיון לדלג מעליו"

        # בצע את המהלך
        self.board.set(from_pos, 0)
        self.board.set(over_pos, 0)
        self.board.set(to_pos, 1)

        # שמור בהיסטוריה
        move = (from_pos, over_pos, to_pos)
        self.move_history.append(move)
        self.move_log.append(move)
        self.redo_stack.clear()  # ניקוי ערימת ה-redo לאחר מהלך חדש
        return True, from_pos, over_pos, to_pos

    def undo(self):
        if not self.move_history:
            return
        last_move = self.move_history.pop()
        from_pos, over_pos, to_pos = last_move

        self.board.set(from_pos, 1)
        self.board.set(over_pos, 1)
        self.board.set(to_pos, 0)

        self.redo_stack.append(last_move)
        self.move_log.pop()

    def redo(self):
        if not self.redo_stack:
            return
        move_to_redo = self.redo_stack.pop()
        from_pos, over_pos, to_pos = move_to_redo

        self.board.set(from_pos, 0)
        self.board.set(over_pos, 0)
        self.board.set(to_pos, 1)

        self.move_history.append(move_to_redo)
        self.move_log.append(move_to_redo)

    def reset(self):
        self.board.reset()
        self.move_history.clear()
        self.redo_stack.clear()
        self.move_log.clear()

    def is_win(self):
        return self.board.count_pegs() == 1 and self.board.get((3, 3)) == 1

    def is_game_over(self):
        if self.is_win():
            return False
        # בדיקה פשוטה, ניתן לממש בדיקה מלאה של כל המהלכים האפשריים
        return self.board.count_pegs() > 1 and len(self.get_all_possible_moves()) == 0

    def get_all_possible_moves(self):
        moves = []
        for r_from, c_from in Board.LEGAL_POSITIONS:
            if self.board.get((r_from, c_from)) == 1:
                for dr, dc in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                    r_to, c_to = r_from + dr, c_from + dc
                    if (r_to, c_to) in Board.LEGAL_POSITIONS and self.board.get((r_to, c_to)) == 0:
                        r_over, c_over = r_from + dr // 2, c_from + dc // 2
                        if self.board.get((r_over, c_over)) == 1:
                            moves.append(((r_from, c_from), (r_to, c_to)))
        return moves

    def export_move_log(self):
        return self.move_log


# =============================================================================
# 🗂  Session-state
# =============================================================================
if "game" not in st.session_state:
    st.session_state.game = Game()
if "sel" not in st.session_state:  # פיון מסומן (או None)
    st.session_state.sel = None
if "error_message" not in st.session_state:
    st.session_state.error_message = None
if "success_message" not in st.session_state:
    st.session_state.success_message = None


# =============================================================================
# 🎨 פונקציות עזר ל-UI
# =============================================================================
def handle_click(row, col):
    """מטפל בלחיצה על משבצת בלוח"""
    game = st.session_state.game
    click = (row, col)
    st.session_state.error_message = None
    st.session_state.success_message = None

    # שלב ראשון – בחירת פיון
    if st.session_state.sel is None:
        if game.board.get(click) == 1:
            st.session_state.sel = click
        # אין צורך ב-rerun, הלולאה תמשיך וה-UI יתעדכן ממילא
    # שלב שני – ניסיון קפיצה
    else:
        from_pos = st.session_state.sel
        to_pos = click

        # אם המשתמש לוחץ שוב על אותו פיון, בטל את הבחירה
        if from_pos == to_pos:
            st.session_state.sel = None
            return

        applied, _, _, _ = game.apply_move(from_pos, to_pos)
        st.session_state.sel = None  # איפוס הבחירה בכל מקרה
        if applied:
            st.session_state.success_message = f"המהלך בוצע: {from_pos} ➝ {to_pos}"
        else:
            st.session_state.error_message = "מהלך לא חוקי"


def get_peg_display(r, c):
    """מחזיר את התו (אימוג'י) להצגה במשבצת"""
    pos = (r, c)
    is_selected = (st.session_state.sel == pos)

    if st.session_state.game.board.get(pos) == 1:
        return "🔵" if is_selected else "🟢"  # פיון מסומן / פיון רגיל
    elif st.session_state.game.board.get(pos) == 0:
        return "⚪"  # חור
    else:
        return ""  # מחוץ ללוח


# =============================================================================
# 🏷  כותרת
# =============================================================================
st.title("🧠 מחשבת – Peg Solitaire")

# =============================================================================
# 🎨  הצגת הלוח עם כפתורים
# =============================================================================
st.markdown("<div style='direction: ltr;'>", unsafe_allow_html=True)
# יצירת רשת של עמודות
grid_cols = st.columns(7)
for r in range(7):
    # כל שורה בתוך העמודה המתאימה
    with grid_cols[r]:
        for c in range(7):
            pos = (c, r)  # שים לב להיפוך, כי אנחנו בונים עמודה-עמודה
            display_char = get_peg_display(c, r)

            if display_char:
                st.button(
                    display_char,
                    key=f"btn_{c}_{r}",
                    on_click=handle_click,
                    args=(c, r)
                )
            else:
                # משבצת ריקה כדי לשמור על מבנה הרשת
                st.markdown("<div style='height: 39px;'></div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# 🔘  כפתורי שליטה
# =============================================================================
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("↩️ Undo", disabled=not st.session_state.game.move_history, use_container_width=True):
        st.session_state.game.undo()
        st.session_state.error_message = None
        st.session_state.success_message = None
        st.rerun()
with col2:
    if st.button("↪️ Redo", disabled=not st.session_state.game.redo_stack, use_container_width=True):
        st.session_state.game.redo()
        st.session_state.error_message = None
        st.session_state.success_message = None
        st.rerun()
with col3:
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.game.reset()
        st.session_state.sel = None
        st.session_state.error_message = None
        st.session_state.success_message = None
        st.rerun()

# =============================================================================
# ℹ️  סטטוס ולוג
# =============================================================================
game = st.session_state.game

# הצגת הודעות הצלחה או שגיאה
if st.session_state.success_message:
    st.success(st.session_state.success_message)
if st.session_state.error_message:
    st.error(st.session_state.error_message)

peg_cnt = game.board.count_pegs()
move_cnt = len(game.move_log)

if game.is_win():
    st.success("🎉 ניצחון! נשאר פיון יחיד במרכז.")
elif game.is_game_over():
    st.warning("🛑 אין מהלכים חוקיים – נסה שוב.")
else:
    st.info(f"פינים: {peg_cnt} | מהלכים: {move_cnt}")

with st.expander("📜 לוג מהלכים"):
    if not game.move_log:
        st.write("אין מהלכים עדיין.")
    for i, (f, o, t) in enumerate(game.export_move_log(), 1):
        st.write(f"{i:2}: {f} ➝ {t} (דרך {o})")