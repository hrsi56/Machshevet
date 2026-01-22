# We assume the Board class from the previous step is available
# class Board: ...
from typing import List, Tuple
import PegActionSpace
from Board import Board
import copy


class Game:
    """
    Game logic for Peg Solitaire.
    Holds a Board, manages move legality, applying moves, undo/redo, and game conditions.
    The move log is derived directly from the move history to prevent sync issues.
    """

    DIRECTIONS = [(-2, 0), (2, 0), (0, -2), (0, 2)]

    def __init__(self, board=None, reward_fn=None):
        self.board = board.copy() if board else Board()
        # move_history is the single source of truth.
        # Format: [(from, to, over, board_state_BEFORE_move)]
        self.move_history = []
        self.redo_stack = []
        self.last_move = None
        self.reward_fn = reward_fn
        self.custom_metadata = {}

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

    def apply_move(self, action: Tuple[int, int, int]):
        """
        הגרסה החדשה: מקבלת פעולה בפורמט קנוני.
        """
        # נשתמש בפונקציית עזר כדי לתרגם את הפעולה למהלך
        from_pos, to_pos = PegActionSpace().action_to_move(action)  # נניח שיש לנו גישה ל-ActionSpace

        legal, over_pos = self.is_legal_move(from_pos, to_pos)
        if not legal:
            return False, 0.0, self.is_game_over(), {"reason": "Illegal move"}

        board_before = self.board.copy()
        self.move_history.append((from_pos, to_pos, over_pos, board_before))
        self.redo_stack.clear()

        self.board.set(from_pos, 0)
        self.board.set(over_pos, 0)
        self.board.set(to_pos, 1)

        self.last_move = (from_pos, over_pos, to_pos)

        done = self.is_game_over()
        reward = self.reward_fn(self.board.copy(), self.last_move, done) if self.reward_fn else 0.0
        info = {"last_move": self.last_move, "done": done}

        return True, reward, done, info

    def undo(self):
        if not self.move_history:
            return False

        (from_pos, to_pos, over_pos, board_before) = self.move_history.pop()
        self.redo_stack.append((from_pos, to_pos, over_pos, self.board.copy()))
        self.board = board_before

        # The last move is now the one at the top of the history stack.
        self.last_move = self.move_history[-1][:3] if self.move_history else None
        return True

    def redo(self):
        if not self.redo_stack:
            return False

        board_before_redo = self.board.copy()
        (from_pos, to_pos, over_pos, board_after_move) = self.redo_stack.pop()
        self.board = board_after_move

        self.move_history.append((from_pos, to_pos, over_pos, board_before_redo))

        self.last_move = (from_pos, over_pos, to_pos)
        return True

    # NEW: The log is now a read-only property derived from the true history.
    def export_move_log(self):
        """
        Returns the move log, built directly from the move history.
        This ensures the log is always in sync with the actual state.
        Format of a move is (from_pos, to_pos, over_pos).
        """
        # The move data is the first 3 elements of each tuple in move_history
        return [item[:3] for item in self.move_history]

    # --- Other methods remain the same ---
    def is_game_over(self):
        return len(self.get_legal_moves()) == 0

    def is_win(self):
        return self.board.count_pegs() == 1 and self.board.get((3, 3)) == 1

    def reset(self, board=None):
        self.board = board.copy() if board else Board()
        self.move_history.clear()
        self.redo_stack.clear()
        self.last_move = None
        self.custom_metadata.clear()

    def get_state(self):
        return self.board.copy()

    def set_state(self, state):
        if isinstance(state, Board):
            self.board = state.copy()
        elif isinstance(state, dict):
            new_board = Board()
            new_board.set_state(state)
            self.board = new_board
        else:
            raise ValueError("Invalid state type for set_state")
        self.move_history.clear()
        self.redo_stack.clear()
        self.last_move = None

    def __hash__(self):
        return hash(self.board)

    def __eq__(self, other):
        return isinstance(other, Game) and self.board == other.board

    def get_custom_metadata(self, key, default=None):
        return self.custom_metadata.get(key, default)

    def set_custom_metadata(self, key, value):
        self.custom_metadata[key] = value

    def clone_state(self):
        """
        Creates a deep copy of the current game state.
        Essential for MCTS simulations.
        """
        # copy.deepcopy היא דרך פשוטה אך לעיתים איטית.
        # עבור האובייקטים שלנו היא מספיק טובה ויעילה.
        return copy.deepcopy(self)

    def get_final_reward(self, is_win: bool) -> float:
        """
        מחזיר את התגמול הסופי האמיתי של המשחק,
        בהתאם ללוגיקה של פונקציית התגמול שבחרנו.
        """
        if is_win:
            return 1000.0  # תואם את פונקציית התגמול שלנו
        else:
            # במצב הפסד, התגמול שלנו היה תלוי במספר החיילים.
            # לצורך MCTS, ערך שלילי קבוע וגדול הוא מספיק טוב.
            return -200.0

    def get_legal_moves(self) -> List[Tuple[int, int, int]]:
        """
        הגרסה החדשה: מחזירה את כל המהלכים החוקיים בפורמט קנוני
        של (שורה, עמודה, אינדקס-כיוון).
        """
        moves = []
        for r_from in range(7):
            for c_from in range(7):
                from_pos = (r_from, c_from)
                if self.board.get(from_pos) != 1:
                    continue

                # נעבור על 4 הכיוונים האפשריים
                for dir_idx, d in enumerate(self.DIRECTIONS):
                    to_pos = (from_pos[0] + d[0], from_pos[1] + d[1])

                    is_legal, _ = self.is_legal_move(from_pos, to_pos)
                    if is_legal:
                        # הוספת הפעולה בפורמט הקנוני (r, c, dir_idx)
                        moves.append((r_from, c_from, dir_idx))
        return moves
