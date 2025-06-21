from __future__ import annotations
from typing import Dict, Tuple, List, Union
import numpy as np
from matplotlib import pyplot as plt
from torch import autograd, autocast, GradScaler
import pickle
import tkinter as tk
from pathlib import Path

Pos = Tuple[int, int]

class Board:
    """
    7×7 Peg-Solitaire cross board.
    ייצוג — 1 = פיון, 0 = חור / תא-חוץ (ערוץ המסכה יבדיל חוקיות).
    """

    __slots__ = ("state",)

    # --- גאומטריה -------------------------------------------------------
    LEGAL_POSITIONS: List[Pos] = [
        (r, c) for r in range(7) for c in range(7)
        if (2 <= r <= 4) or (2 <= c <= 4)
    ]
    LEGAL_MASK: np.ndarray = np.zeros((7, 7), dtype=np.float32)
    for _r, _c in LEGAL_POSITIONS:
        LEGAL_MASK[_r, _c] = 1.0

    TOTAL_PEGS: int = 32          # (מרכז ריק)

    # -------------------------------------------------------------------
    def __init__(self) -> None:
        self.state: Dict[Pos, int] = {p: 1 for p in self.LEGAL_POSITIONS}
        self.state[(3, 3)] = 0

    # ---------------- בסיסיים ------------------------------------------
    def reset(self) -> None:
        for p in self.LEGAL_POSITIONS:
            self.state[p] = 1
        self.state[(3, 3)] = 0

    def get(self, pos: Pos) -> int | None: return self.state.get(pos)
    def set(self, pos: Pos, val: int) -> None:
        if pos not in self.LEGAL_POSITIONS or val not in (0, 1):
            raise ValueError(f"illegal {pos=}/{val=}")
        self.state[pos] = val

    def all_pegs(self) -> List[Pos]:  return [p for p,v in self.state.items() if v]
    def all_holes(self) -> List[Pos]: return [p for p,v in self.state.items() if not v]
    def count_pegs(self) -> int:      return sum(self.state.values())

    # ---------------- array / dict --------------------------------------
    def as_array(self) -> np.ndarray:
        arr = np.zeros((7, 7), dtype=np.float32)
        for p in self.LEGAL_POSITIONS:
            arr[p] = float(self.state[p])
        return arr
    get_state = as_array

    def set_state(self, data: Union[np.ndarray, Dict[Pos, int]]) -> None:
        if isinstance(data, dict):
            for p in self.LEGAL_POSITIONS:
                self.state[p] = int(data.get(p, 0))
        else:
            arr = np.asarray(data, dtype=np.float32)
            if arr.shape != (7, 7):
                raise ValueError("state array must be (7,7)")
            for p in self.LEGAL_POSITIONS:
                self.state[p] = 1 if arr[p] == 1 else 0

    def to_dict(self) -> Dict[Pos, int]: return self.state.copy()

    # ---------------- NN-קידוד -----------------------------------------
    def encode_observation(self) -> np.ndarray:
        arr = self.as_array()
        peg_cnt   = arr.sum()
        removed   = (self.TOTAL_PEGS - peg_cnt) / self.TOTAL_PEGS
        remaining = peg_cnt / self.TOTAL_PEGS

        obs = np.zeros((7, 7, 4), dtype=np.float32)
        obs[:, :, 0] = arr
        obs[:, :, 1] = removed
        obs[:, :, 2] = remaining
        obs[:, :, 3] = self.LEGAL_MASK
        return obs

    # ---------------- אוגמנטציה (סימטריות) -----------------------------
    @staticmethod
    def augment_observation(obs: np.ndarray, mode: str = "random") -> np.ndarray:
        augs = []
        for rot in range(4):
            rot_img = np.rot90(obs, k=rot, axes=(0, 1))
            augs.append(rot_img)
            augs.append(np.flip(rot_img, axis=1))
        if mode == "all":
            return np.stack(augs)
        if mode == "random":
            return augs[np.random.randint(8)]
        return obs

    # ---------------- שירות/השוואה -------------------------------------
    def copy(self) -> "Board":
        b = Board()
        b.state = self.state.copy()
        return b
    clone = copy

    def __hash__(self) -> int: return hash(tuple(sorted(self.state.items())))
    def __eq__(self, o: object) -> bool:
        return isinstance(o, Board) and o.state == self.state

    def __str__(self) -> str:
        arr = self.as_array()
        rows = [" ".join("●" if arr[r,c]==1 else "◯" if arr[r,c]==0 else " "
                         for c in range(7)) for r in range(7)]
        return "\n".join(rows)

from typing import Tuple, List, Dict, Optional, Callable, Union

Pos    = Tuple[int, int]           # (row, col)
Action = Tuple[int, int, int]      # (row, col, dir-idx)

import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 1)

    def forward(self, x):
        """
        x: טנזור בגודל [B, 4, 7, 7] הנגזר מ־Board.encode_observation()
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.contiguous().flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)  # -> [B]


from typing import Callable, List, Tuple, Optional, Dict, Union

Pos    = Tuple[int, int]
Action = Tuple[int, int, int]


class Game:
    """
    Peg-Solitaire game engine (7×7 cross).
    • תומך ב־RL (clone, reward, actions)
    • תיעוד מהלכים, Undo/Redo
    • reward shaping עם פוטנציאל נוירוני
    """

    DIRECTIONS: List[Pos] = [(-2, 0), (2, 0), (0, -2), (0, 2)]  # up, down, left, right

    def __init__(
        self,
        board: Optional["Board"] = None,
        reward_fn: Optional[Callable[[bool, float, float], float]] = None,
        potential_fn: Optional[Callable[["Board"], float]] = None,
    ) -> None:
        self.board: "Board" = board.copy() if board else Board()
        self.move_history: List[Tuple[Pos, Pos, Pos, "Board"]] = []
        self.redo_stack  : List[Tuple[Pos, Pos, Pos, "Board"]] = []
        self.last_move   : Optional[Tuple[Pos, Pos, Pos]] = None
        self.move_log    : List[Tuple[Pos, Pos, Pos]] = []

        self.potential_fn = potential_fn or (lambda b: -b.count_pegs())
        self.reward_fn = self._default_reward
        self.custom_metadata: Dict[str, object] = {}

    @staticmethod
    def _middle(a: Pos, b: Pos) -> Pos:
        return ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)

    def is_legal_move(self, src: Pos, dst: Pos) -> Tuple[bool, Optional[Pos]]:
        if src not in Board.LEGAL_POSITIONS or dst not in Board.LEGAL_POSITIONS:
            return False, None
        dr, dc = dst[0] - src[0], dst[1] - src[1]
        if (abs(dr), abs(dc)) not in ((2, 0), (0, 2)):
            return False, None
        over = self._middle(src, dst)
        if self.board.get(src) == 1 and self.board.get(over) == 1 and self.board.get(dst) == 0:
            return True, over
        return False, None

    def get_legal_moves(self) -> List[Tuple[Pos, Pos, Pos]]:
        moves: List[Tuple[Pos, Pos, Pos]] = []
        for src in self.board.all_pegs():
            for di, dj in self.DIRECTIONS:
                dst = (src[0] + di, src[1] + dj)
                ok, over = self.is_legal_move(src, dst)
                if ok:
                    moves.append((src, dst, over))
        return moves

    def get_legal_actions(self) -> List[Action]:
        acts: List[Action] = []
        for src in self.board.all_pegs():
            for d_idx, (di, dj) in enumerate(self.DIRECTIONS):
                dst = (src[0] + di, src[1] + dj)
                if self.is_legal_move(src, dst)[0]:
                    acts.append((src[0], src[1], d_idx))
        return acts

    def is_legal_action(self, a: Action) -> bool:
        r, c, d = a
        dr, dc  = self.DIRECTIONS[d]
        return self.is_legal_move((r, c), (r + dr, c + dc))[0]

    def _apply(self, src: Pos, dst: Pos, over: Pos) -> None:
        self.board.set(src, 0)
        self.board.set(over, 0)
        self.board.set(dst, 1)

    # בתוך המחלקה Game

    def apply_move(self, src: Pos, dst: Pos) -> Tuple[bool, float, bool, Dict]:
        """
        Apply a legal move (src → dst), return success, reward, done, info.
        This version is optimized for Potential-Based Reward Shaping.
        """
        ok, over = self.is_legal_move(src, dst)
        if not ok:
            # אם המהלך לא חוקי, החזר קנס קבוע.
            return False, -1.0, self.is_game_over(), {"reason": "illegal move"}

        # --- התחלה: יישום PBRS יעיל ---

        # 1. חשב את הפוטנציאל של הלוח *לפני* ביצוע המהלך.
        potential_before = self.potential_fn(self.board)

        # שמור את מצב הלוח הנוכחי לצורך פונקציונליות ה-Undo.
        before_board_state = self.board.copy()
        self.move_history.append((src, dst, over, before_board_state))
        self.redo_stack.clear()

        # 2. בצע את המהלך בפועל על הלוח.
        self._apply(src, dst, over)
        self.last_move = (src, over, dst)
        self.move_log.append(self.last_move)

        # 3. בדוק אם המשחק הסתיים וחשב את הפוטנציאל *אחרי* המהלך.
        done = self.is_game_over()
        potential_after = self.potential_fn(self.board)

        # 4. חשב את התגמול באמצעות פונקציית התגמול החדשה,
        #    שמקבלת את ערכי הפוטנציאל ישירות.
        #    שים לב: אנחנו מניחים ש-self.reward_fn עודכנה לקבל חתימה זו.
        reward = self.reward_fn(done=done, potential_before=potential_before, potential_after=potential_after)

        # --- סוף: יישום PBRS יעיל ---

        info = {
            "last_move": self.last_move,
            "done": done,
            "is_solved": self.is_win(),
            "num_pegs": self.board.count_pegs(),
        }
        return True, reward, done, info


    def apply_action(self, a: Action) -> Tuple[bool, float, bool, Dict]:
        r, c, d = a
        dr, dc  = self.DIRECTIONS[d]
        return self.apply_move((r, c), (r + dr, c + dc))

    def undo(self) -> bool:
        if not self.move_history:
            return False
        src, dst, over, before = self.move_history.pop()
        self.redo_stack.append((src, dst, over, self.board.copy()))
        self.board = before
        if self.move_log:
            self.move_log.pop()
        self.last_move = self.move_history[-1][:3] if self.move_history else None
        return True

    def redo(self) -> bool:
        if not self.redo_stack:
            return False
        src, dst, over, before = self.redo_stack.pop()
        self.move_history.append((src, dst, over, before))
        self.board = before.copy()
        self._apply(src, dst, over)
        self.last_move = (src, over, dst)
        self.move_log.append(self.last_move)
        return True

    def is_game_over(self) -> bool:
        return not self.get_legal_moves()

    def is_win(self) -> bool:
        return self.board.count_pegs() == 1 and self.board.get((3, 3)) == 1

    is_solved = is_win  # alias

    def num_pegs(self) -> int:
        return self.board.count_pegs()

    def reset(self, board: Optional["Board"] = None) -> None:
        self.board = board.copy() if board else Board()
        self.move_history.clear()
        self.redo_stack.clear()
        self.last_move = None
        self.move_log.clear()
        self.custom_metadata.clear()

    def clone_state(self) -> "Game":
        return Game(
            board=self.board.copy(),
            reward_fn=self.reward_fn,
            potential_fn=self.potential_fn,
        )

    def export_move_log(self) -> List[Tuple[Pos, Pos, Pos]]:
        return self.move_log.copy()

    def set_state(self, state: Union["Board", Dict[Pos, int]]) -> None:
        if isinstance(state, Board):
            self.board = state.copy()
        elif isinstance(state, dict):
            self.board.set_state(state)
        else:
            raise TypeError("state must be Board or dict")
        self.move_history.clear()
        self.redo_stack.clear()
        self.last_move = None

    def get_custom_metadata(self, k: str, default=None):
        return self.custom_metadata.get(k, default)

    def set_custom_metadata(self, k: str, v):
        self.custom_metadata[k] = v

    # Game class

    def _default_reward(
            self,
            done: bool,
            potential_before: float,
            potential_after: float
    ) -> float:
        """
        PBRS מלא: ΔΦ בכל צעד + בונוס/קנס סופי חד-פעמי.
        """
        # ייתכן שתרצה להתאים את הגאמא בהתאם לבעיה ולרשת.
        # ערך נמוך יותר יכול לעודד פתרונות מהירים יותר.
        # ערך גבוה יותר יכול לעודד למידה לטווח ארוך.
        gamma = 0.995  # נסה ערך מעט גבוה יותר כדי לתת יותר משקל לעתיד

        # בונוסים/קנסות סופיים. ייתכן שצריך להגדיל/להקטין אותם.
        # אם הסוכן לא מצליח לפתור, הגדל את term_win או הקטן את term_loss.
        # אם הסוכן מוצא קיצורי דרך לא רצויים, הקטן את term_win או הגדל את term_loss.
        term_win = 10.0  # בונוס משמעותי על ניצחון
        term_loss = -5.0  # קנס משמעותי על הפסד

        # Δ-potential עבור כל צעד
        shaped = gamma * potential_after - potential_before

        if not done:
            return shaped

        # פרס סופי – מתווסף ל-ΔΦ של הצעד האחרון בלבד
        return shaped + (term_win if self.is_win() else term_loss)

    def __str__(self) -> str:
        parts = [str(self.board)]
        if self.last_move:
            parts.append(f"Last move: {self.last_move}")
        return "\n".join(parts)

import numpy as np
from typing import Callable, List, Tuple, Optional, Union

# טיפוסים משותפים
Pos    = Tuple[int, int]
Action = Tuple[int, int, int]


# ------------------------------------------------------------------
# 🗺️  STRATEGIC MAPS  (7×7 English board – ערכים ניתנים לכיול)
# ------------------------------------------------------------------
import numpy as np
from typing import Callable, List, Optional, Tuple, Union

CENTRALITY_WEIGHTS = np.array(
	[[0.0, 0.0, 0.10, 0.10, 0.10, 0.0, 0.0],
	 [0.0, 0.10, 0.20, 0.30, 0.20, 0.10, 0.0],
	 [0.10, 0.20, 0.40, 0.50, 0.40, 0.20, 0.10],
	 [0.10, 0.30, 0.50, 1.00, 0.50, 0.30, 0.10],
	 [0.10, 0.20, 0.40, 0.50, 0.40, 0.20, 0.10],
	 [0.0, 0.10, 0.20, 0.30, 0.20, 0.10, 0.0],
	 [0.0, 0.0, 0.10, 0.10, 0.10, 0.0, 0.0]], dtype=np.float32)

PAGODA_VALUES = np.array(
	[[0, 0, 1, 2, 1, 0, 0],
	 [0, 1, 2, 3, 2, 1, 0],
	 [1, 2, 3, 4, 3, 2, 1],
	 [2, 3, 4, 5, 4, 3, 2],
	 [1, 2, 3, 4, 3, 2, 1],
	 [0, 1, 2, 3, 2, 1, 0],
	 [0, 0, 1, 2, 1, 0, 0]], dtype=np.float32)

DIRS_JUMP = np.array([[-2, 0], [2, 0], [0, -2], [0, 2]], dtype=np.int8)
CORNER_POSITIONS = [(0, 3), (3, 0), (3, 6), (6, 3)]
EDGE_MASK = ((CENTRALITY_WEIGHTS < 0.25) & (CENTRALITY_WEIGHTS > 0)).astype(np.float32)


# ------------------------------------------------------------------
#  PegSolitaireEnv  (הגרסה המקורית שלך + פוטנציאל מובנה)
# ------------------------------------------------------------------
class PegSolitaireEnv:
    """
    Gym-lite environment for Peg-Solitaire (7×7 cross).
    Observation : ndarray (7,7,4);  Action : (row,col,dir-idx)
    """

    BOARD_SIZE = 7
    TOTAL_PEGS = 32

    # -------- ctor -------- #
    def __init__(
        self,
        board_cls,
        game_cls,
        reward_fn: Optional[Callable[["Board", Tuple, bool], float]] = None,
        potential_fn: Optional[Callable[["Board"], float]] = None,
    ) -> None:
        self._board_cls = board_cls
        self._game_cls  = game_cls
        self.board_mask = board_cls.LEGAL_MASK.astype(np.float32).copy()

        # אם לא סופקה Φ(s) חיצונית – השתמש בזו המובנית
        potential_fn = self._calculate_potential

        self.game = game_cls(board_cls(), reward_fn=reward_fn, potential_fn=potential_fn)
        self._potential_fn = potential_fn
        self.done = False

    # -------- פוטנציאל מובנה -------- #
    # --------------------------------------------------------------
    #  Φ(s)  –  פוטנציאל מרובד (תיקון bounds-safe ל-isolation)
    # --------------------------------------------------------------
    # ------------------------------------------------------------------
    #  GLOBAL STRATEGIC MAPS  (7×7 English board) – tuned for accuracy
    # ------------------------------------------------------------------
    import numpy as np

    # ==============================================================
    #   Φ(s)  – High-accuracy layered potential
    # ==============================================================
    def _calculate_potential(self, board) -> float:
        arr  = board.as_array().astype(np.float32)          # 1 / 0
        mask = self.board_mask
        peg_cnt = int(arr.sum())

        # --- φ0 : peg count (fewer ≈ better) -----------------------
        phi_num = -peg_cnt                                    # linear

        # --- φ1 : centrality (average per peg) --------------------
        phi_centr = (arr * CENTRALITY_WEIGHTS * mask).sum() / max(1, peg_cnt)

        # --- φ2 : pagoda resource ---------------------------------
        phi_pagoda = (arr * PAGODA_VALUES * mask).sum()

        # --- φ3 : penalties – isolated / edge / corner ------------
        if peg_cnt == 0:
            phi_iso_edge = 0.0
        else:
            rc = np.argwhere(arr == 1)                         # N×2
            reachable = np.zeros(len(rc), dtype=bool)

            for dr, dc in DIRS_JUMP:
                mid = rc + (dr//2, dc//2)
                dst = rc + (dr, dc)

                valid = (
                    (dst[:,0] >= 0) & (dst[:,0] < 7) &
                    (dst[:,1] >= 0) & (dst[:,1] < 7)
                )
                if not valid.any():
                    continue
                idx = np.where(valid)[0]

                dst_ok    = mask[dst[idx,0], dst[idx,1]] == 1
                mid_ok    = mask[mid[idx,0], mid[idx,1]] == 1
                has_peg   = arr[mid[idx,0], mid[idx,1]] == 1
                empty_dst = arr[dst[idx,0], dst[idx,1]] == 0    # ← החלק הקריטי

                reachable[idx] |= (dst_ok & mid_ok & has_peg & empty_dst)

            isolated = int((~reachable).sum())
            corners  = int(sum(arr[r,c] for r,c in CORNER_POSITIONS))
            edges    = int((arr * EDGE_MASK).sum())

            phi_iso_edge = -(5.0*isolated + 2.0*corners + 1.0*edges)

        # --- weighted sum (hyper-params tuned for accuracy) -------
        w0, w1, w2, w3 = 1.2, 0.65, 1.0, 0.85
        return w0*phi_num + w1*phi_centr + w2*phi_pagoda + w3*phi_iso_edge

    # ------------------------------------------------------------------
    # כל שאר השיטות שלך נשארו ללא שינוי – העתקנו ככתבן וכלשונן
    # ------------------------------------------------------------------
    def reset(self, state: Optional[Union[dict, "Board"]] = None) -> Tuple[np.ndarray, dict]:
        if state is None:
            self.game.reset()
        else:
            self.game.set_state(state)
        self.done = False
        return self.encode_observation(), {"num_pegs": self.game.board.count_pegs()}

    def step(self, action: Tuple[int, int, int]) -> Tuple[np.ndarray, float, bool, dict]:
        if self.done:
            raise RuntimeError("step() called after episode finished — call reset().")

        if self.game.is_legal_action(action):
            _, reward, self.done, _ = self.game.apply_action(action)
        else:
            reward = -1.0
            self.done = self.game.is_game_over()

        obs = self.encode_observation()
        info = {
            "num_pegs":  self.game.board.count_pegs(),
            "is_solved": self.game.is_solved(),
        }
        return obs, reward, self.done, info

    def get_legal_actions(self) -> List:
        return self.game.get_legal_actions()

    def get_legal_action_mask(self, action_space_size: int, to_idx: Callable) -> np.ndarray:
        mask = np.zeros(action_space_size, dtype=np.float32)
        for a in self.get_legal_actions():
            mask[to_idx(a)] = 1.0
        return mask

    def encode_observation(self) -> np.ndarray:
        arr     = self.game.board.as_array().astype(np.float32)
        peg_cnt = arr.sum()
        removed = (self.TOTAL_PEGS - peg_cnt) / self.TOTAL_PEGS
        remain  = peg_cnt / self.TOTAL_PEGS

        obs = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE, 4), dtype=np.float32)
        obs[..., 0] = arr
        obs[..., 1] = removed
        obs[..., 2] = remain
        obs[..., 3] = self.board_mask
        return obs

    def clone_state(self, state=None) -> "PegSolitaireEnv":
        clone = PegSolitaireEnv(
            self._board_cls,
            self._game_cls,
            reward_fn=self.game.reward_fn,
            potential_fn=self._potential_fn,
        )
        clone.reset(state or self.game.board.to_dict())
        return clone

    @staticmethod
    def augment_observation(obs: np.ndarray, mode: str = "random"):
        augs: List[np.ndarray] = []
        for rot in range(4):
            rot_img = np.rot90(obs, k=rot, axes=(0, 1))
            for flip in (False, True):
                augs.append(np.flip(rot_img, axis=1) if flip else rot_img)
        if mode == "all":
            return augs
        if mode == "random":
            return augs[np.random.randint(8)]
        return obs

    @staticmethod
    def augment_action(action, rot: int = 0, flip: bool = False):
        row, col, d = action
        dirs = [(-2, 0), (2, 0), (0, -2), (0, 2)]
        dr, dc = dirs[d]
        tgt = (row + dr, col + dc)

        for _ in range(rot):
            row, col = col, 6 - row
            tgt = (tgt[1], 6 - tgt[0])

        if flip:
            col, tgt = 6 - col, (tgt[0], 6 - tgt[1])

        diff = (tgt[0] - row, tgt[1] - col)
        return row, col, dirs.index(diff)

    def render(self, mode="human"):
        print(self.game.board)

from typing import Tuple




# Assuming ResidualBlock is defined elsewhere and works correctly
class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class PegSolitaireNet(nn.Module):
    """
    ResNet-10 dual-head network with MHSA-based Policy + MC Dropout for Uncertainty.
    Input  : (B,4,7,7)
    Output : π_logits (B,n_actions), v (B,) ∈ [-1,1]
    """
    def __init__(
        self,
        n_actions: int,
        channels: int = 64,
        n_res_blocks: int = 6,
        hidden_value: int = 128,
        dropout_p: float = 0.1,
        use_layernorm: bool = True,
        device: str | torch.device = "mps",
    ) -> None:
        super().__init__()
        assert channels % 4 == 0, "channels must be divisible by number of attention heads (4)"
        self.device = torch.device(device)

        # ----- Stem -----
        self.conv_in = nn.Conv2d(4, channels, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(channels)

        # ----- Residual Tower -----
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(n_res_blocks)])

        # ===== Policy Head with Multi-Head Self-Attention =====
        self.attn_mhsa = nn.MultiheadAttention(embed_dim=channels, num_heads=4, batch_first=True)
        self.policy_norm = nn.LayerNorm(channels)
        self.pol_fc = nn.Linear(channels, n_actions)

        # ===== Value Head =====
        self.val_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.val_bn = nn.BatchNorm2d(2)
        self.val_fc1 = nn.Linear(2 * 7 * 7, hidden_value)
        self.val_fc2 = nn.Linear(hidden_value, 1)
        self.dropout = nn.Dropout(p=dropout_p)
        self.ln_value = nn.LayerNorm(hidden_value) if use_layernorm else nn.Identity()

        self.to(self.device)

    # --------------------------------------------------------------------- #
    def _policy_head(self, features: torch.Tensor) -> torch.Tensor:
        """
        Multi-Head Self-Attention over spatial features → logits (B, n_actions)
        """
        B, C, H, W = features.shape  # Expected: (B, C, 7, 7)
        x = features.flatten(2).transpose(1, 2)           # (B, 49, C)
        attn_out, _ = self.attn_mhsa(x, x, x)              # (B, 49, C)
        pooled = attn_out.mean(dim=1)                      # (B, C)
        pooled = self.policy_norm(pooled)                  # (B, C)
        return self.pol_fc(pooled)                         # (B, n_actions)

    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args    : x (B,4,7,7)
        Returns : π_logits (B,n_actions), v (B,) ∈ [-1,1]
        """
        x = x.to(self.device)
        x = F.relu(self.bn_in(self.conv_in(x)))            # (B, C, 7, 7)
        x = self.body(x)                                   # (B, C, 7, 7)

        # ----- Policy Head -----
        pi_logits = self._policy_head(x)                   # (B, n_actions)

        # ----- Value Head -----
        v = F.relu(self.val_bn(self.val_conv(x)))          # (B, 2, 7, 7)
        v = v.reshape(v.shape[0], -1)                      # (B, 98)
        v = self.dropout(F.relu(self.val_fc1(v)))          # (B, hidden)
        v = self.ln_value(v)
        v = torch.tanh(self.val_fc2(v)).squeeze(-1)        # (B,)

        return pi_logits, v

from collections import deque


import heapq

import numpy as np
import random
import torch
from typing import List, Tuple

import numpy as np
import torch
import heapq
import itertools
import random
from typing import List, Tuple, Optional

import numpy as np
import random
import torch
import heapq
import itertools
from typing import List, Tuple, Optional

class ReplayBuffer:
    """
    Prioritized Experience Replay Buffer (PER) with priority-based forgetting.
    Efficient heapq-based implementation. Compatible API.
    Stores (observation, policy π, value z).
    """
    def __init__(self, max_size: int = 50_000, alpha: float = 0.6):
        self.max_size = max_size
        self.alpha = alpha
        self.counter = itertools.count()
        self.heap: List[Tuple[float, int, Tuple[np.ndarray, np.ndarray, float]]] = []

    def push(self, sample: Tuple[np.ndarray, np.ndarray, float], priority: Optional[float] = None) -> None:
        if priority is None:
            priority = max([abs(x[0]) for x in self.heap], default=1.0)
        entry = (float(priority), next(self.counter), sample)
        heapq.heappush(self.heap, entry)
        if len(self.heap) > self.max_size:
            heapq.heappop(self.heap)

    def sample(self, batch_size: int) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        # Take a snapshot for consistent indices
        heap_list = list(self.heap)
        priorities = np.array([abs(x[0]) for x in heap_list], dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum() + 1e-8

        idxs = np.random.choice(len(heap_list), batch_size, p=probs)
        return [heap_list[i][2] for i in idxs]

    def sample_as_tensors(
        self,
        batch_size: int,
        device: str = "mps"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        heap_list = list(self.heap)
        priorities = np.array([abs(x[0]) for x in heap_list], dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum() + 1e-8

        idxs = np.random.choice(len(heap_list), batch_size, p=probs)
        obs = torch.tensor(np.stack([heap_list[i][2][0] for i in idxs]), dtype=torch.float32, device=device).permute(0, 3, 1, 2).contiguous()
        pi  = torch.tensor(np.stack([heap_list[i][2][1] for i in idxs]), dtype=torch.float32, device=device).contiguous()
        z   = torch.tensor(np.array([heap_list[i][2][2] for i in idxs]), dtype=torch.float32, device=device).contiguous()
        return obs, pi, z, idxs

    def update_priorities(self, indices: np.ndarray, new_priorities: np.ndarray) -> None:
        """
        Update priorities by building a new heap with updated priorities.
        This avoids ValueError and keeps all samples fresh.
        """
        heap_list = list(self.heap)
        for heap_idx, new_prio in zip(indices, new_priorities):
            old_prio, count, sample = heap_list[heap_idx]
            heap_list[heap_idx] = (float(new_prio), count, sample)
        # Rebuild the heap in place (this is O(N), but done rarely)
        self.heap = heap_list
        heapq.heapify(self.heap)

    def __len__(self) -> int:
        return len(self.heap)


# הערה: מחלקת Node נשארה ללא שינוי, לכן אין צורך להציגה שוב.
# אני מניח שהיא זמינה במודול.

class _Node:
    """Internal MCTS node (PUCT)."""
    __slots__ = ("prior", "children", "visit", "value_sum")

    def __init__(self, prior: float) -> None:
        self.prior: float = prior
        self.children: Dict[int, "_Node"] = {}
        self.visit: int = 0
        self.value_sum: float = 0.0

    @property
    def value(self) -> float:
        return self.value_sum / self.visit if self.visit > 0 else 0.0


class MCTS:
    """
    PUCT-based Monte-Carlo Tree Search, AlphaZero style.
    עובד על-פי ביקורי צמתים, תומך ב-root Dirichlet noise, אופטימלי לביצועים.
    """

    def __init__(
            self,
            env,
            model: "PegSolitaireNet",
            action_space,
            sims: int = 400,
            c_puct: float = 1.5,
            root_noise: bool = True,
            dirichlet_alpha: float = 0.3,
            noise_ratio: float = 0.25,
            device: str | torch.device = "mps",
    ) -> None:
        self.env = env
        self.model = model
        self.action_space = action_space
        self.sims = sims
        self.c_puct = c_puct
        self.root_noise = root_noise
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_ratio = noise_ratio
        self.device = torch.device(device)

    def run(self, root_obs: np.ndarray, tau: float = 1.0, entropy_threshold: float = 0.5) -> np.ndarray:
        """
        DS-MCTS (Dynamic Stopping MCTS) AlphaZero-style implementation.
        Stops simulations dynamically based on entropy of visit distribution.
        """
        root = self._expand(
            obs=root_obs,
            legal=self.env.get_legal_actions(),
            add_noise=self.root_noise,
        )

        for sim in range(self.sims):
            env_cpy = self.env.clone_state()
            node = root
            path = [root]
            done = False

            # -------- Selection -------- #
            while node.children and not done:
                act_idx, node = self._select(node)
                _, _, done, _ = env_cpy.step(self.action_space.from_index(act_idx))
                path.append(node)

            # -------- Expansion & Evaluation -------- #
            if not done:
                obs = env_cpy.encode_observation()
                node.children = self._expand(obs, env_cpy.get_legal_actions()).children
                v = self._evaluate(obs)
            else:
                v = 1.0 if env_cpy.game.is_win() else -1.0

            # -------- Backpropagation -------- #
            for n in path:
                n.visit += 1
                n.value_sum += v

            # -------- Dynamic stopping based on entropy -------- #
            visits = np.array([child.visit for child in root.children.values()])
            probs = visits / visits.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-8))

            if entropy < entropy_threshold and sim > (self.sims * 0.3):
                # Allow a minimum number of simulations (e.g., 30%) before dynamic stopping
                break

        # -------- Compute final policy π -------- #
        sorted_children = sorted(root.children.items())
        visits = np.array([child.visit for _, child in sorted_children])
        indices = np.array([idx for idx, _ in sorted_children])

        if tau == 0.0:
            pi = np.zeros_like(visits, dtype=np.float32)
            pi[np.argmax(visits)] = 1.0
        else:
            log_visits = np.log(visits + 1e-8) / tau
            pi = np.exp(log_visits - np.max(log_visits))
            pi /= pi.sum()

        full_pi = np.zeros(len(self.action_space), dtype=np.float32)
        full_pi[indices] = pi

        return full_pi


    @torch.no_grad()
    def _evaluate(self, obs: np.ndarray) -> float:
        x = torch.tensor(obs, dtype=torch.float32, device=self.device) \
            .permute(2, 0, 1).unsqueeze(0).contiguous()
        _, v = self.model(x)
        return float(v.item())

    def _expand(
            self,
            obs: np.ndarray,
            legal: List["Action"],
            add_noise: bool = False,
    ) -> "_Node":
        x = torch.tensor(obs, dtype=torch.float32, device=self.device) \
            .permute(2, 0, 1).unsqueeze(0).contiguous()
        logits, _ = self.model(x)
        probs = torch.softmax(logits, -1).cpu().detach().numpy().reshape(-1)

        mask = self.action_space.legal_action_mask(legal)
        probs *= mask

        # נרמול מחדש לאחר המסוך
        if probs.sum() > 1e-8:
            probs /= probs.sum()
        else:
            probs = mask / mask.sum()

        if add_noise:
            legal_indices = np.where(mask == 1.0)[0]
            if len(legal_indices) > 0:
                noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_indices))
                noisy_probs = np.zeros_like(probs)
                noisy_probs[legal_indices] = noise
                probs = (1 - self.noise_ratio) * probs + self.noise_ratio * noisy_probs

        node = _Node(prior=1.0)
        for a in legal:
            idx = self.action_space.to_index(a)
            node.children[idx] = _Node(prior=float(probs[idx]))
        return node

    def _select(self, node: "_Node") -> Tuple[int, "_Node"]:
        total_sqrt_visits = np.sqrt(node.visit)
        best_score, best_idx = -np.inf, -1
        for idx, child in node.children.items():
            q_value = child.value
            u_value = self.c_puct * child.prior * total_sqrt_visits / (1 + child.visit)
            score = q_value + u_value
            if score > best_score:
                best_score, best_idx = score, idx
        return best_idx, node.children[best_idx]


# ------------------------------------------------------------------ #
#                               Agent                                #
# ------------------------------------------------------------------ #

from typing import  Tuple

Pos = Tuple[int, int]
Action = Tuple[int, int, int]

# agent.py  – industrial-grade version (steps 2-5 applied)
import time
from typing import Dict, Tuple, List, Optional, Callable



try:
    import mlflow
    _HAVE_MLFLOW = True
except ImportError:        # אפשר להריץ גם בלי MLflow (יוצג אזהרה בלוג)
    _HAVE_MLFLOW = False
    import logging
    logging.warning("MLflow not found – running without experiment tracking.")

# --------------------------------------------------------------------------- #
# ---------------------------   ANALYSIS TOOLS   ---------------------------- #
# --------------------------------------------------------------------------- #
class AgentAnalyzer:
    """
    מקליט החלטות במהלך self-play ומפיק דו״חות.
    משמש לדיבוג מתקדם (שלב 5).
    """
    def __init__(self, action_space):
        self.action_space = action_space
        self.reset()

    def reset(self) -> None:
        self.decision_log: List[Dict] = []

    def record(self, obs: np.ndarray, act_idx: int, value_est: float,
               legal_idx: List[int]) -> None:
        self.decision_log.append({
            "obs": obs, "action_idx": act_idx, "value": value_est,
            "legal_idx": legal_idx
        })

    # דוח קצר; ניתן להרחיב בהתאם לצורך
    def summary(self) -> Dict[str, float]:
        from collections import Counter
        actions = [d["action_idx"] for d in self.decision_log]
        act_counter = Counter(actions)
        return {
            "total_moves": len(actions),
            "unique_actions": len(act_counter),
            "top_action": act_counter.most_common(1)[0] if actions else None,
        }

# --------------------------------------------------------------------------- #
# ------------------------------   AGENT   ---------------------------------- #
# --------------------------------------------------------------------------- #
# ====================================================================== #
#                              agent.py                                  #
# ====================================================================== #
import os, time, random, math
from collections import defaultdict
from typing import List, Dict, Tuple, Callable, Optional



try:
    import mlflow                           # optional
    _HAVE_MLFLOW = True
except ImportError:
    _HAVE_MLFLOW = False

# type aliases
Pos    = Tuple[int, int]
Action = Tuple[int, int, int]

# ---------------------------------------------------------------------- #
class AgentAnalyzer:
    """אוסף נתונים במהלך self-play לצורך דיבוג וסטטיסטיקות."""
    def __init__(self, action_space):
        self.action_space = action_space
        self.reset()

    def reset(self):
        self.records: List[Dict] = []

    def log(self, obs, act_idx, value_est, legal_idx):
        self.records.append(dict(obs=obs,
                                 action_idx=int(act_idx),
                                 value=float(value_est),
                                 legal=legal_idx))

    def summary(self) -> Dict[str, float]:
        if not self.records:
            return {"total_moves": 0}
        counts = defaultdict(int)
        for r in self.records:
            counts[r["action_idx"]] += 1
        most_common = max(counts.items(), key=lambda kv: kv[1])
        return {
            "total_moves": len(self.records),
            "unique_actions": len(counts),
            "top_action": self.action_space.from_index(most_common[0]),
            "top_freq": most_common[1],
        }

# ---------------------------------------------------------------------- #
# ========================================================= #
#                           Agent                           #
# ========================================================= #
class Agent:
    """
    AlphaZero-style Agent for Peg-Solitaire.
    - Self-play with MCTS
    - Training (OneCycleLR, no AMP)
    - MLflow optional
    - Checkpointing, Analyzer
    """

    # ------------------------- ctor ------------------------ #
    def __init__(
        self,
        env: "PegSolitaireEnv",
        model: "PegSolitaireNet",
        action_space: "PegSolitaireActionSpace",
        buffer: "ReplayBuffer",
        sims: int = 33,
        device: str | torch.device = "mps",
        keep_history: bool = False,
        mlflow_experiment: str | None = None,
        ckpt_dir: str = "checkpoints",
    ) -> None:

        self.device = torch.device(device)
        self.env, self.model = env, model.to(self.device)
        self.action_space, self.buffer = action_space, buffer

        self.mcts = MCTS(
            env, self.model, action_space,
            sims=sims, root_noise=True,
            device=self.device
        )

        # history / debug
        self.keep_history = keep_history
        self.episodes: List[Dict] = [] if keep_history else []
        self.stats = {"episodes": 0, "success": 0, "avg_moves": 0.0}
        self._global_step = 0

        # infra
        self.ckpt_dir = Path(ckpt_dir);  self.ckpt_dir.mkdir(exist_ok=True)
        self.analyzer = AgentAnalyzer(action_space)

        # MLflow
        self._mlflow_ctx = None
        if _HAVE_MLFLOW and mlflow_experiment:
            mlflow.set_experiment(mlflow_experiment)
            self._mlflow_ctx = mlflow.start_run()

    # --------------------- helpers ------------------------ #
    def _transform_policy(self, π: np.ndarray, rot: int, flip: bool) -> np.ndarray:
        """Rotate/flip policy vector to match board augmentation."""
        if rot == 0 and not flip:
            return π
        new_π = np.zeros_like(π)
        for idx, prob in enumerate(π):
            if prob < 1e-9:  # skip near-zero
                continue
            a     = self.action_space.from_index(idx)
            a_aug = self.env.augment_action(a, rot=rot, flip=flip)
            new_π[self.action_space.to_index(a_aug)] = prob
        return new_π

    # ------------------ self-play episode ------------------ #
    def self_play_episode(self, augment: bool = True) -> None:
        obs, _ = self.env.reset()
        done = False
        moves = 0
        total_reward = 0.0
        states, policies = [], []

        self.analyzer.reset()
        episode_record = {
            "moves": [],
            "reward": 0.0,
            "solved": False,
            "moves_len": 0,
        }

        while not done:
            moves += 1
            tau = 1.0 if moves < 10 else 0.05

            # Compute policy using MCTS
            π = self.mcts.run(obs, tau=tau)

            # Mask policy probabilities based on legal actions
            legal_actions = self.env.get_legal_actions()
            legal_indices = [self.action_space.to_index(a) for a in legal_actions]
            π_masked = π[legal_indices]
            π_masked /= np.sum(π_masked) + 1e-8

            # Choose action probabilistically based on masked policy
            chosen_index = np.random.choice(legal_indices, p=π_masked)
            action = self.action_space.from_index(chosen_index)

            # Store state and policy for replay buffer
            states.append(obs.copy())
            policies.append(π.copy())

            # Analyzer logging
            with torch.no_grad():
                tensor_obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                tensor_obs = tensor_obs.permute(2, 0, 1).unsqueeze(0).contiguous()
                _, v_est = self.model(tensor_obs)

            self.analyzer.log(obs, chosen_index, v_est.item(), legal_indices)

            # Execute chosen action
            obs, reward, done, _ = self.env.step(action)
            total_reward = reward
            episode_record["moves"].append(action)

        # Record the episode outcome
        episode_record.update({
            "reward": total_reward,
            "solved": total_reward > 0,
            "moves_len": moves,
        })

        if self.keep_history:
            episode_record["analyzer"] = self.analyzer.summary()
            self.episodes.append(episode_record)

        # Update stats
        self.stats["episodes"] += 1
        self.stats["avg_moves"] = (
                (self.stats["avg_moves"] * (self.stats["episodes"] - 1) + moves)
                / self.stats["episodes"]
        )
        if total_reward > 0:
            self.stats["success"] += 1

        # Push states, policies, and rewards into the buffer
        for state, policy in zip(states, policies):
            if augment:
                for rot in range(4):
                    for flip in (False, True):
                        augmented_state = np.rot90(state, k=rot, axes=(0, 1))
                        if flip:
                            augmented_state = np.flip(augmented_state, axis=1)
                        augmented_policy = self._transform_policy(policy, rot, flip)
                        self.buffer.push((augmented_state, augmented_policy, total_reward))
            else:
                self.buffer.push((state, policy, total_reward))


    # ------------------ checkpoint / eval ------------------ #
    def _save_ckpt(self, tag: str):
        torch.save({
            "step": self._global_step,
            "model": self.model.state_dict(),
            "stats": self.stats,
        }, self.ckpt_dir / f"agent_{tag}.pt")

    def _quick_eval(self, n_ep: int = 5) -> float:
        solved = 0
        for _ in range(n_ep):
            obs, _ = self.env.reset(); done = False
            while not done:
                π = self.mcts.run(obs, tau=0.0)
                action = self.action_space.from_index(int(np.argmax(π)))
                obs, r, done, _ = self.env.step(action)
            solved += (r > 0)
        return solved / n_ep

    # ------------------------- train ----------------------- #
    def train(
            self,
            batch: int = 256,
            epochs: int = 1,
            lr: float = 1e-3,
            log_cb: Optional[Callable[[Dict], None]] = None,
    ) -> None:
        """
        מאמן את הרשת תוך שימוש ב־OneCycleLR, כולל MLflow logging ו־Checkpointing.
        תומך ב־PER עם Importance Sampling Weights.
        """
        if len(self.buffer) < batch:
            return

        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr,
            steps_per_epoch=max(1, len(self.buffer) // batch),
            epochs=epochs
        )

        start = time.time()
        if not hasattr(self, "_sims0"):
            self._sims0 = self.mcts.sims  # 33 כברירת-מחדל

        growth = 1.18  # 18 % בכל Epoch
        cap = 384  # תקרה קשיחה
        # -----------------------------------------------------------------

        for epoch in range(epochs):
            # --- הגדלת מספר הסימולציות ---
            self.mcts.sims = min(
                int(self._sims0 * (growth ** epoch)),
                cap
            )

            # ==== דגימה עם אינדקסים וחשבון IS weights ====
            obs_t, pi_t, z_t, indices = self.buffer.sample_as_tensors(batch, device=self.device)

            # חשב את חשיבות הדגימות עבור PER-IS
            # snapshot של priorities
            heap_list = list(self.buffer.heap)
            priorities = np.array([abs(x[0]) for x in heap_list], dtype=np.float32)
            probs = priorities ** self.buffer.alpha
            probs /= probs.sum() + 1e-8

            sample_probs = probs[indices]
            # חישוב IS weights: w_i = (1/N * 1/p_i) ** beta  -- נורמליזציה
            beta = getattr(self.buffer, 'beta', 0.4)
            N = len(self.buffer)
            is_weights = (N * sample_probs) ** (-beta)
            is_weights /= is_weights.max() + 1e-8
            is_weights = torch.tensor(is_weights, dtype=torch.float32, device=self.device).unsqueeze(1)  # (B,1)

            opt.zero_grad()
            logits, v_pred = self.model(obs_t)

            # --- Policy Loss (לא צריך IS weight, משקלות על value בלבד) ---
            loss_pol = F.kl_div(
                F.log_softmax(logits, dim=-1),
                pi_t,
                reduction="batchmean"
            )
            # --- Value Loss + IS Weights ---
            td_error = v_pred - z_t
            loss_val = ((td_error ** 2) * is_weights.squeeze()).mean()
            loss = loss_pol + loss_val

            loss.backward()
            opt.step()

            # ==== עדכון priorities (PER) ====
            td_err = (v_pred.detach() - z_t).abs().cpu().numpy()
            td_err = td_err / (td_err.max() + 1e-6)
            self.buffer.update_priorities(indices, td_err + 1e-6)

            sched.step()
            self._global_step += 1

            # Logging
            log = {
                "step": self._global_step,
                "epoch": epoch,
                "loss": loss.item(),
                "loss_pol": loss_pol.item(),
                "loss_val": loss_val.item(),
                "lr": sched.get_last_lr()[0],
            }
            if log_cb:
                log_cb(log)

            if self._mlflow_ctx:
                mlflow.log_metrics({
                    "loss": loss.item(),
                    "loss_pol": loss_pol.item(),
                    "loss_val": loss_val.item(),
                    "lr": log["lr"],
                }, step=self._global_step)

            # Checkpointing and periodic evaluation
            if (epoch + 1) % max(1, epochs // 5) == 0:
                success_rate = self._quick_eval()
                if self._mlflow_ctx:
                    mlflow.log_metric("success_rate", success_rate, step=self._global_step)
                self._save_ckpt(f"epoch_{epoch + 1:02d}")

        if self._mlflow_ctx:
            mlflow.log_metric("train_time_sec", time.time() - start)
            mlflow.end_run()

            # ------------------------------------------------------------------ #
#                        Action-Space helper                         #
# ------------------------------------------------------------------ #
class PegSolitaireActionSpace:
    DIRS = 4  # ↑ ↓ ← →

    def __init__(self, board_mask: np.ndarray) -> None:
        if board_mask.shape != (7, 7):
            raise ValueError("board_mask must be shape (7,7)")

        self.valid_cells: List[Pos] = [
            (r, c) for r in range(7) for c in range(7) if board_mask[r, c] == 1
        ]
        self.actions: List[Action] = [
            (r, c, d) for r, c in self.valid_cells for d in range(self.DIRS)
        ]
        self._to_idx: Dict[Action, int] = {a: i for i, a in enumerate(self.actions)}

    def to_index(self, a: Action) -> int:
        if a not in self._to_idx:
            raise ValueError(f"Invalid action: {a}")
        return self._to_idx[a]

    def from_index(self, idx: int) -> Action:
        if idx < 0 or idx >= len(self.actions):
            raise IndexError(f"Action-index {idx} out of range 0‥{len(self.actions)-1}")
        return self.actions[idx]

    def legal_action_mask(self, legal: List[Action]) -> np.ndarray:
        mask = np.zeros(len(self.actions), dtype=np.float32)
        for a in legal:
            mask[self._to_idx[a]] = 1.0
        return mask

    def is_valid_action(self, a: Action) -> bool:
        return a in self._to_idx

    def sample_legal(self, legal: List[Action]) -> Action:
        return random.choice(legal)

    def action_str(self, a: Action) -> str:
        d_names = ["↑", "↓", "←", "→"]
        return f"{a[0]},{a[1]} {d_names[a[2]]}"

    @property
    def size(self) -> int:
        return len(self.actions)

    def __len__(self) -> int:
        return len(self.actions)

    def __iter__(self):
        return iter(self.actions)

    def __repr__(self) -> str:
        return f"<ActionSpace size={len(self)}>"






# -------- הגדרות כלליות -------- #
AGENT_PATH = Path("peg_agent.pt")
HISTORY_PATH = Path("episode_history.pkl")
TRAIN_EPISODES = 5
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🔧 Using device: {DEVICE}")


# -------- שמירה / טעינה -------- #
def save_agent(agent: Agent, path: Path = AGENT_PATH) -> None:
    data = {
        "state_dict": agent.model.state_dict(),
        "n_actions": len(agent.action_space),
        "sims": agent.mcts.sims,
    }
    torch.save(data, path)
    print(f"✅ Agent saved to: {path.resolve()}")


def load_agent(path: Path = AGENT_PATH) -> Agent:
    if not path.exists():
        raise FileNotFoundError(f"No agent found at {path}")

    ckpt = torch.load(path, map_location=DEVICE)
    env = PegSolitaireEnv(Board, Game)
    asp = PegSolitaireActionSpace(env.board_mask)
    model = PegSolitaireNet(ckpt["n_actions"], device=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    print("📦 Agent loaded successfully.")
    return Agent(env, model, asp, ReplayBuffer(), sims=ckpt.get("sims", 33), device=DEVICE, keep_history=True)


# -------- אימון / הרצה -------- #
def train_new_agent(episodes: int = TRAIN_EPISODES) -> Agent:
    env = PegSolitaireEnv(Board, Game)
    asp = PegSolitaireActionSpace(env.board_mask)
    model = PegSolitaireNet(len(asp), device=DEVICE)
    buffer = ReplayBuffer()
    agent = Agent(env, model, asp, buffer, sims=33, device=DEVICE, keep_history=True)

    print("🚀 Starting training loop ...")
    for ep in range(1, episodes + 1):
        print(ep)
        agent.self_play_episode()

        if len(buffer) >= 256:
            agent.train(batch=256, epochs=1, lr=1e-3)

        if ep % 1 == 0:
            s = agent.stats
            print(f"  ↳ ep {ep:4d}/{episodes} | "
                  f"buffer={len(buffer):5d} | "
                  f"success={s['success']}/{s['episodes']} | "
                  f"avg_moves={s['avg_moves']:.1f}")

    save_agent(agent)
    if agent.keep_history:
        with open(HISTORY_PATH, "wb") as f:
            pickle.dump(agent.episodes, f)
        print(f"🧾 Episode history saved to: {HISTORY_PATH.resolve()}")

    return agent



PLOTS_DIR = Path("plots")


# ------------------ Plotting ------------------ #
def generate_plots():
    if not HISTORY_PATH.exists():
        print("ℹ️ No history to plot.")
        return
    hist = pickle.load(open(HISTORY_PATH, "rb"))
    if not hist:
        print("ℹ️ Empty history.")
        return

    ep = np.arange(1, len(hist) + 1)
    rew = np.array([h["reward"] for h in hist])
    mv = np.array([len(h["moves"]) for h in hist])
    suc = np.cumsum([h["solved"] for h in hist]) / ep

    PLOTS_DIR.mkdir(exist_ok=True)
    def plot_series(x, y, title, ylabel, fname):
        plt.figure(figsize=(9,3))
        plt.plot(x,y); plt.title(title); plt.xlabel("Episode"); plt.ylabel(ylabel)
        plt.grid(True); plt.tight_layout()
        plt.savefig(PLOTS_DIR/fname)
        plt.close()

    plot_series(ep, rew, "Reward per Episode", "Reward","reward.png")
    plot_series(ep, mv,  "Moves per Episode", "Moves", "moves.png")
    plot_series(ep, suc, "Cumulative Success-Rate", "Success-rate", "success.png")
    print(f"✅ Plots saved. Final success rate: {suc[-1]*100:.2f}%")

from tkinter import messagebox
class PegSolitaireGUI(tk.Frame):
    """GUI אינטראקטיבי כולל רמזים מה-Agent (פועל גם ללא Agent)."""

    # ——— קבועי עיצוב ———
    CELL, R, PAD = 60, 22, 16
    PEG, HOLE, OUTL, HILITE = "#FFD600", "#202020", "#333", "#42A5F5"
    SUGGEST, BG = "#00C853", "#eeeeee"
    BAR_W, BAR_H = 160, 16

    def __init__(self, master, game: Game, agent: Agent | None = None) -> None:
        super().__init__(master, bg=self.BG)
        self.game, self.agent = game, agent
        self.sel: tuple[int, int] | None = None  # חור נבחר
        self.hint: tuple[tuple[int, int], tuple[int, int]] | None = None  # (src,dst)

        side = 7 * self.CELL + 2 * self.PAD
        self.canvas = tk.Canvas(self, width=side, height=side,
                                bg=self.BG, highlightthickness=0)
        self.canvas.pack()

        # ——— פס עליון ———
        top = tk.Frame(self, bg=self.BG)
        top.pack(pady=4, fill="x")
        self.status = tk.Label(top, font=("Arial", 14), bg=self.BG, anchor="w")
        self.status.pack(side="left", expand=True, fill="x")
        self.bar = tk.Canvas(top, width=self.BAR_W, height=self.BAR_H,
                             bg=self.BG, highlightthickness=0)
        self.bar.pack(side="right", padx=6)

        # ——— כפתורים ———
        btns = tk.Frame(self, bg=self.BG);
        btns.pack()
        for txt, cmd in [("\u21a9 Undo", self.on_undo),
                         ("\u21aa Redo", self.on_redo),
                         ("\u21bb Reset", self.on_reset),
                         ("\U0001f916 Hint", self.on_hint)]:
            tk.Button(btns, text=txt, command=cmd).pack(side="left", padx=3)

        # ——— לוג מהלכים ———
        self.log = tk.Listbox(self, width=42, height=6, font=("Consolas", 11))
        self.log.pack(pady=(8, 0))

        # ——— קיצורי מקשים ———
        self.canvas.bind("<Button-1>", self.on_click)
        master.bind("<Control-z>", lambda e: self.on_undo())
        master.bind("<Control-y>", lambda e: self.on_redo())

        self.redraw()

    # ------------------------------------------------------------------ #
    #                            ציור לוח                                #
    # ------------------------------------------------------------------ #
    def _xy(self, pos: tuple[int, int]) -> tuple[int, int]:
        """המרת (row,col) לקואורדינטות קנבס."""
        return (self.PAD + pos[1] * self.CELL + self.CELL // 2,
                self.PAD + pos[0] * self.CELL + self.CELL // 2)

    def redraw(self) -> None:
        self.canvas.delete("all")

        # פגים / חורים
        for pos in Board.LEGAL_POSITIONS:
            x, y = self._xy(pos)
            fill = self.PEG if self.game.board.get(pos) == 1 else self.HOLE
            width = 3 if pos == self.sel else 1
            outline = self.HILITE if pos == self.sel else self.OUTL
            self.canvas.create_oval(x - self.R, y - self.R, x + self.R, y + self.R,
                                    fill=fill, outline=outline, width=width)

        # מהלכים חוקיים מהחור הנבחר
        if self.sel:
            for s, d, _ in self.game.get_legal_moves():
                if s == self.sel:
                    x, y = self._xy(d)
                    self.canvas.create_oval(x - self.R // 2, y - self.R // 2,
                                            x + self.R // 2, y + self.R // 2,
                                            outline=self.HILITE, width=3)

        # רמז
        if self.hint:
            src, dst = self.hint
            x1, y1 = self._xy(src);
            x2, y2 = self._xy(dst)
            self.canvas.create_line(x1, y1, x2, y2,
                                    fill=self.SUGGEST, width=5, arrow=tk.LAST)

        self._update_status()
        self._update_log()
        self._update_bar()

    # ------------------------------------------------------------------ #
    #                          פס הערך                                    #
    # ------------------------------------------------------------------ #
    def _update_bar(self) -> None:
        v = 0.0
        if self.agent:
            obs = self.game.board.encode_observation()
            t = torch.tensor(obs, dtype=torch.float32, device=self.agent.device)
            t = t.permute(2, 0, 1).unsqueeze(0).contiguous()
            with torch.no_grad():
                _, v_out = self.agent.model(t)
                v = float(v_out)

        frac = (v + 1) / 2  # ↦ [0,1]
        length = int(frac * self.BAR_W)
        col = "#d50000" if v < -0.3 else "#9e9e9e" if v < 0.3 else "#00c853"
        self.bar.delete("all")
        self.bar.create_rectangle(0, 0, length, self.BAR_H, fill=col, width=0)
        self.bar.create_rectangle(0, 0, self.BAR_W, self.BAR_H, outline="#555")

    # ------------------------------------------------------------------ #
    #                     אינטראקציה עם לוח                               #
    # ------------------------------------------------------------------ #
    def on_click(self, e) -> None:
        pos = ((e.y - self.PAD) // self.CELL,
               (e.x - self.PAD) // self.CELL)
        if pos not in Board.LEGAL_POSITIONS:
            return

        if self.sel is None and self.game.board.get(pos) == 1:
            self.sel = pos
        elif self.sel and pos != self.sel:
            success, _ = self.game.apply_move(self.sel, pos)[:2]
            if success:
                self.sel = self.hint = None
        else:
            self.sel = None
        self.redraw()

    # קיצורי פעולות
    def on_undo(self):
        self._call(self.game.undo)

    def on_redo(self):
        self._call(self.game.redo)

    def on_reset(self):
        self._call(self.game.reset)

    def _call(self, fn):
        if not fn():
            return
        self.sel = self.hint = None
        self.redraw()

    # ------------------------------------------------------------------ #
    #                            רמז מה-Agent                             #
    # ------------------------------------------------------------------ #
    def _move_to_action(self, move: tuple) -> tuple[int, int, int]:
        """
        המרה מ-(src,dst,mid) לפורמט (row,col,dir).

        • תומכת גם ב-DIRECTIONS באורך 1 וגם באורך 2.
        """
        src, dst, _ = move
        dr, dc = dst[0] - src[0], dst[1] - src[1]

        for d, (drow, dcol) in enumerate(Game.DIRECTIONS):
            # אם Game.DIRECTIONS = (±1,0/0,±1) → קפיצה היא 2*δ
            if (dr, dc) == (2 * drow, 2 * dcol):
                return (src[0], src[1], d)
            # אם Game.DIRECTIONS = (±2,0/0,±2) → קפיצה שווה בדיוק ל־δ
            if (dr, dc) == (drow, dcol):
                return (src[0], src[1], d)

        raise ValueError(f"Cannot map move {move} to action – check DIRECTIONS.")

    def on_hint(self) -> None:
        if not self.agent:
            messagebox.showinfo("Hint", "No agent loaded.")
            return
        if self.game.is_game_over():
            self.status.config(text="Game over.")
            return

        # הפעל את הרשת ישירות (מהיר בהרבה מ-MCTS)
        obs = self.game.board.encode_observation()
        t = torch.tensor(obs, dtype=torch.float32, device=self.agent.device)
        t = t.permute(2, 0, 1).unsqueeze(0).contiguous()
        with torch.no_grad():
            logits, _ = self.agent.model(t)
            π = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        legal_moves = self.game.get_legal_moves()
        if not legal_moves:
            self.status.config(text="No legal moves.")
            self.hint = None
            return

        # מיפוי legal_move → action
        legal_actions = [self._move_to_action(m) for m in legal_moves]
        legal_indices = [self.agent.action_space.to_index(a) for a in legal_actions]

        π_masked = np.zeros_like(π)
        π_masked[legal_indices] = π[legal_indices]
        if π_masked.sum() == 0:
            self.status.config(text="Agent unsure.")
            self.hint = None
            return

        best_idx = int(np.argmax(π_masked))
        best_action = self.agent.action_space.from_index(best_idx)
        dr, dc = Game.DIRECTIONS[best_action[2]]
        self.hint = ((best_action[0], best_action[1]),
                     (best_action[0] + dr, best_action[1] + dc))
        self.redraw()

    # ------------------------------------------------------------------ #
    #                סטטוס / לוג מהלכים                                   #
    # ------------------------------------------------------------------ #
    def _update_status(self) -> None:
        if self.game.is_win():
            text = "Victory! Single peg in center."
        elif self.game.is_game_over():
            text = f"Game Over in {len(self.game.move_log)} moves."
        else:
            text = f"Pegs: {self.game.board.count_pegs()} | Moves: {len(self.game.move_log)}"
        self.status.config(text=text)

    def _update_log(self) -> None:
        self.log.delete(0, tk.END)
        for i, (s, _, d) in enumerate(self.game.move_log, 1):
            self.log.insert(tk.END, f"{i:2}: {s} → {d}")


# ------------------------------------------------------------------ #
#                                Main                                #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    # טען Agent אם קיים, אחרת הרץ ללא Agent (GUI יעבוד, רק ללא רמזים)
    if AGENT_PATH.exists():
        agent = load_agent()
    else :
        agent = train_new_agent()
    generate_plots()
    root = tk.Tk()
    root.title("Peg-Solitaire AI")
    PegSolitaireGUI(root, Game(), agent).pack()
    root.mainloop()