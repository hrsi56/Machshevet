from __future__ import annotations
from typing import Dict, Tuple, List, Union
import numpy as np
from torch import autograd, autocast, GradScaler

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

def make_potential_fn(model: ValueNetwork, device='cpu'):
    def potential(board: "Board") -> float:
        obs = board.encode_observation()  # [7,7,4]
        x = torch.tensor(obs, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)  # -> [1,4,7,7]
        with torch.no_grad():
            return model(x).item()
    return potential
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
        self.reward_fn = reward_fn or self._default_reward
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
        Calculates the reward using Potential-Based Reward Shaping (PBRS).
        """
        # ------------------ פרמטרים ------------------
        gamma = 0.99  # פקטור היוון, צריך להיות זהה לזה של המטרה (z)
        win_reward = 1.0  # תגמול על ניצחון
        loss_penalty = -1.0  # קנס על הפסד (סיום משחק ללא פתרון)
        step_penalty = -0.01  # קנס קטן על כל צעד למניעת משחקים אינסופיים (אופציונלי)
        # -------------------------------------------

        if done:
            return win_reward if self.is_win() else loss_penalty

        # נוסחת PBRS: R = F(s, a, s') = r(s,a) + gamma * Φ(s') - Φ(s)
        # כאן, r(s,a) הוא תגמול הצעד המיידי (step_penalty)
        reward_shaping = gamma * potential_after - potential_before
        return step_penalty + reward_shaping


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


class PegSolitaireEnv:
    """
    Gym-lite environment for Peg-Solitaire (7×7 cross).

    Observation : ndarray (7, 7, 4) —
        ch0 = peg / hole,
        ch1 = fraction removed,
        ch2 = fraction remaining,
        ch3 = legal-mask.

    Action      : (row, col, dir-idx)  where dir-idx∈{0:↑, 1:↓, 2:←, 3:→}
    """

    BOARD_SIZE = 7
    TOTAL_PEGS = 32

    def __init__(
        self,
        board_cls,
        game_cls,
        reward_fn: Optional[Callable[["Board", Tuple, bool], float]] = None,
        potential_fn: Optional[Callable[["Board"], float]] = None,
    ) -> None:
        self._board_cls = board_cls
        self._game_cls  = game_cls
        self.board_mask = board_cls.LEGAL_MASK.copy()

        self.game = game_cls(board_cls(), reward_fn=reward_fn, potential_fn=potential_fn)
        self._potential_fn = potential_fn
        self.done = False

    def reset(self, state: Optional[Union[dict, "Board"]] = None) -> Tuple[np.ndarray, dict]:
        if state is None:
            self.game.reset()
        else:
            self.game.set_state(state)
        self.done = False
        return self.encode_observation(), {"num_pegs": self.game.board.count_pegs()}

    def step(self, action: Action) -> Tuple[np.ndarray, float, bool, dict]:
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

    def get_legal_actions(self) -> List[Action]:
        return self.game.get_legal_actions()

    def get_legal_action_mask(self, action_space_size: int, to_idx: Callable[[Action], int]) -> np.ndarray:
        mask = np.zeros(action_space_size, dtype=np.float32)
        for a in self.get_legal_actions():
            mask[to_idx(a)] = 1.0
        return mask

    def encode_observation(self) -> np.ndarray:
        arr = self.game.board.as_array()
        num_pegs = self.game.board.count_pegs()
        removed, remain = (
            (self.TOTAL_PEGS - num_pegs) / self.TOTAL_PEGS,
            num_pegs / self.TOTAL_PEGS,
        )

        obs = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE, 4), dtype=np.float32)
        obs[:, :, 0] = arr
        obs[:, :, 1] = removed
        obs[:, :, 2] = remain
        obs[:, :, 3] = self.board_mask
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
    def augment_observation(obs: np.ndarray, mode: str = "random") -> List[np.ndarray] | np.ndarray:
        augs: List[np.ndarray] = []
        for rot in range(4):
            r = np.rot90(obs, k=rot, axes=(0, 1))
            for flip in (False, True):
                augs.append(np.flip(r, axis=1) if flip else r)
        if mode == "all":
            return augs
        if mode == "random":
            return augs[np.random.randint(8)]
        return obs

    @staticmethod
    def augment_action(action: Action, rot: int = 0, flip: bool = False) -> Action:
        row, col, d = action
        dirs = [(-2, 0), (2, 0), (0, -2), (0, 2)]
        dr, dc = dirs[d]
        tgt = (row + dr, col + dc)

        for _ in range(rot):
            row, col = col, 6 - row
            tgt      = (tgt[1], 6 - tgt[0])

        if flip:
            row, col = row, 6 - col
            tgt      = (tgt[0], 6 - tgt[1])

        diff  = (tgt[0] - row, tgt[1] - col)
        d_new = dirs.index(diff)
        return row, col, d_new

    def render(self, mode: str = "human") -> None:
        print(self.game.board)
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResidualBlock(nn.Module):
    """Conv-BN-ReLU ×2 + skip connection."""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    ResNet-10 dual-head network with Attention + Uncertainty.
    Input  : (B,4,7,7)
    Output : π_logits (B,n_actions), v (B,)∈[-1,1]
    """
    def __init__(
        self,
        n_actions: int,
        channels: int = 64,
        n_res_blocks: int = 10,
        hidden_value: int = 128,
        dropout_p: float = 0.1,
        use_layernorm: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)

        # ----- Stem -----
        self.conv_in = nn.Conv2d(4, channels, kernel_size=3, padding=1, bias=False)
        self.bn_in   = nn.BatchNorm2d(channels)

        # ----- Residual Tower (10 blocks) -----
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(n_res_blocks)])

        # ===== Policy Head with Spatial-Attention =====
        # 1×1 conv -> attention weights (7×7), softmax-ed
        self.attn_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        # Fully-connected: channels → n_actions
        self.pol_fc    = nn.Linear(channels, n_actions)

        # ===== Value Head =====
        self.val_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.val_bn   = nn.BatchNorm2d(2)
        self.val_fc1  = nn.Linear(2 * 7 * 7, hidden_value)
        self.val_fc2  = nn.Linear(hidden_value, 1)
        self.dropout  = nn.Dropout(p=dropout_p)
        self.ln_value = nn.LayerNorm(hidden_value) if use_layernorm else nn.Identity()

        self.to(self.device)

    # --------------------------------------------------------------------- #
    def _policy_head(self, features: torch.Tensor) -> torch.Tensor:
        """
        Spatial-attention over (B,C,7,7) → logits (B,n_actions)
        """
        B, C, H, W = features.shape                       # 7×7 לגודל לוח קלאסי
        # attention weights α_{i} ∈ ℝ^{49}, Σα=1
        # Changed: Use torch.flatten instead of .contiguous().reshape
        attn_logits  = torch.flatten(self.attn_conv(features), start_dim=1)  # (B,49)
        attn_weights = F.softmax(attn_logits, dim=-1).unsqueeze(1)          # (B,1,49)

        # flatten feature map → (B,C,49) ואז average pool עם α
        # Changed: Use torch.flatten instead of .contiguous().reshape
        f_flat   = torch.flatten(features, start_dim=2)                     # (B,C,49)
        f_weight = (f_flat * attn_weights).sum(dim=2)                      # (B,C)

        # linear → n_actions
        return self.pol_fc(f_weight)                                        # (B,n_actions)

    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args : x (B,4,7,7)
        Returns: π_logits (B,n_actions), v (B,)
        """
        x = x.to(self.device)
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.body(x)                                 # (B,C,7,7)

        # ----- Policy -----
        pi_logits = self._policy_head(x)                 # (B,n_actions)

        # ----- Value -----
        v = F.relu(self.val_bn(self.val_conv(x)))        # (B,2,7,7)
        # Changed: Use torch.flatten instead of .contiguous().flatten
        v = torch.flatten(v, start_dim=1)                # flatten
        v = self.dropout(F.relu(self.val_fc1(v)))        # MC-Dropout ≈ uncertainty
        v = self.ln_value(v)
        v = torch.tanh(self.val_fc2(v)).squeeze(-1)      # (B,)

        return pi_logits, v

from collections import deque
import random
import numpy as np
from typing import List, Tuple, Optional


class ReplayBuffer:
    """
    FIFO Replay Buffer לאחסון דגימות Self-Play בפורמט:
        (observation, policy π, value z)
    כאשר:
        - observation : np.ndarray בגודל (7,7,4)
        - π           : np.ndarray של הסתברויות על פעולות
        - z           : float (ערך המשחק: ניצחון/הפסד)

    תמיכה ב־O(1) הוספה, דגימה אקראית.
    """
    def __init__(self, max_size: int = 50_000) -> None:
        self.max_size = max_size
        self.buf: deque[Tuple[np.ndarray, np.ndarray, float]] = deque(maxlen=max_size)

    def push(self, sample: Tuple[np.ndarray, np.ndarray, float]) -> None:
        """
        הוסף דגימה לבאפר. נחתך אוטומטית אם עבר maxlen.
        """
        self.buf.append(sample)

    def sample(self, batch_size: int) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        דגום אצווה רנדומלית בגודל נתון (או פחות אם חסר).
        """
        batch_size = min(batch_size, len(self.buf))
        return random.sample(self.buf, batch_size)

    def sample_as_tensors(
        self,
        batch_size: int,
        device: str = "cpu"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        דגום אצווה והחזר כ־Torch Tensors.
        """
        import torch
        samples = self.sample(batch_size)
        obs, pi, z = zip(*samples)
        obs = torch.tensor(np.stack(obs), dtype=torch.float32, device=device).permute(0, 3, 1, 2)  # [B,4,7,7]
        pi  = torch.tensor(np.stack(pi), dtype=torch.float32, device=device)
        z   = torch.tensor(np.array(z), dtype=torch.float32, device=device)
        return obs, pi, z

    def __len__(self) -> int:
        return len(self.buf)


import torch
import numpy as np
import random
from typing import Dict, Tuple, List, Optional


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
            sims: int = 100,
            c_puct: float = 1.5,
            root_noise: bool = True,
            dirichlet_alpha: float = 0.3,
            noise_ratio: float = 0.25,
            device: str | torch.device = "cpu",
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

    def run(self, root_obs: np.ndarray, tau: float = 1.0) -> np.ndarray:
        """
        מבצע MCTS מהמצב הנתון ומחזיר את מדיניות הפעולה π.
        """
        root = self._expand(
            obs=root_obs,
            legal=self.env.get_legal_actions(),
            add_noise=self.root_noise,
        )

        for _ in range(self.sims):
            # לכל סימולציה משכפלים סביבה חדשה מהשורש.
            env_cpy = self.env.clone_state()
            node = root
            path = [root]
            done = False

            # -------- 1. Selection (בחירה) -------- #
            while node.children and not done:
                act_idx, node = self._select(node)
                obs, reward, done, _ = env_cpy.step(self.action_space.from_index(act_idx))
                path.append(node)

            # -------- 2. Expansion & Evaluation -------- #
            if not done:
                obs = env_cpy.encode_observation()
                node.children = self._expand(obs, env_cpy.get_legal_actions()).children
                v = self._evaluate(obs)
            else:
                # במשחק חד-שחקן: הערך הוא פשוט התגמול של המצב הסופי.
                v = 1.0 if env_cpy.game.is_win() else -1.0

            # -------- 3. Backpropagation -------- #
            # (אין צורך להחליף סימן – חד שחקן)
            for n in path:
                n.visit += 1
                n.value_sum += v

        # ------- חישוב מדיניות היעד π מביקורי הצמתים -------
        visits = np.array([child.visit for idx, child in sorted(root.children.items())])
        indices = np.array([idx for idx in sorted(root.children)])

        if tau == 0.0:  # בחירה דטרמיניסטית
            pi = np.zeros_like(visits, dtype=np.float32)
            if len(visits) > 0:
                pi[np.argmax(visits)] = 1.0
        else:  # דגימה הסתברותית
            log_v = np.log(visits + 1e-8) / tau
            pi = np.exp(log_v - np.max(log_v))
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
        probs = torch.softmax(logits, -1).cpu().detach().numpy().flatten()

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
import random
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict

Pos = Tuple[int, int]
Action = Tuple[int, int, int]

# agent.py  – industrial-grade version (steps 2-5 applied)
import time
from typing import Dict, Tuple, List, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

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
        sims: int = 100,
        device: str | torch.device = "cpu",
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
        done, moves, reward = False, 0, 0.0
        states, policies = [], []
        self.analyzer.reset()
        rec = {"moves": [], "reward": 0.0, "solved": False, "moves_len": 0}

        while not done:
            moves += 1
            tau = 1.0 if moves < 10 else 0.05
            π = self.mcts.run(obs, tau=tau)

            legal_idx = [self.action_space.to_index(a) for a in self.env.get_legal_actions()]
            π_mask = π[legal_idx]; π_mask /= (π_mask.sum() + 1e-8)
            act_idx = int(np.random.choice(legal_idx, p=π_mask))
            action  = self.action_space.from_index(act_idx)

            states.append(obs); policies.append(π)

            # analyzer
            with torch.no_grad():
                t = torch.tensor(obs, dtype=torch.float32, device=self.device)\
                        .permute(2,0,1).unsqueeze(0)
                _, v_est = self.model(t)
            self.analyzer.log(obs, act_idx, v_est.item(), legal_idx)

            obs, reward, done, _ = self.env.step(action)
            rec["moves"].append(action)

        # bookkeeping
        rec.update({"reward": reward, "solved": reward > 0, "moves_len": moves})
        if self.keep_history:
            rec["analyzer"] = self.analyzer.summary()
            self.episodes.append(rec)

        self.stats["episodes"] += 1
        self.stats["avg_moves"] = (self.stats["avg_moves"]*(self.stats["episodes"]-1)+moves)/self.stats["episodes"]
        if reward > 0:
            self.stats["success"] += 1

        # push to buffer
        for s, π in zip(states, policies):
            if augment:
                for rot in range(4):
                    for flip in (False, True):
                        aug_s = np.rot90(s, k=rot, axes=(0,1))
                        if flip: aug_s = np.flip(aug_s, axis=1)
                        aug_π = self._transform_policy(π, rot, flip)
                        self.buffer.push((aug_s, aug_π, reward))
            else:
                self.buffer.push((s, π, reward))

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

        if len(self.buffer) < batch:
            return

        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr,
            steps_per_epoch=max(1, len(self.buffer)//batch),
            epochs=epochs
        )

        # ---------------- training loop (FP32, no AMP) ---------------- #
        start = time.time()
        for ep in range(epochs):
            # הגדל מספר הסימולציות אם רצוי
            self.mcts.sims = int(self.mcts.sims * (1 + ep / max(1, epochs)))

            # ----- דגימה -----
            obs_t, π_t, z_t = self.buffer.sample_as_tensors(batch, device=self.device)
            # sample_as_tensors כבר מחזיר float32; נוודא:
            obs_t = obs_t.float().contiguous()
            π_t = π_t.float().contiguous()
            z_t = z_t.float().contiguous()

            opt.zero_grad()

            logits, v_hat = self.model(obs_t)  # ← FP32
            loss_pol = F.kl_div(
                F.log_softmax(logits, dim=1), π_t,
                reduction="batchmean"
            )
            loss_val = F.mse_loss(v_hat, z_t)
            loss = loss_pol + loss_val  # עדיין FP32, אין .to()

            loss.backward()  # ← גרף נשמר
            opt.step()
            sched.step()

            self._global_step += 1
            # ... (לוג, checkpoint, MLflow) ...


            # logging
            log = {
                "step": self._global_step,
                "epoch": ep,
                "loss": loss.item(),
                "loss_pol": loss_pol.item(),
                "loss_val": loss_val.item(),
                "lr": sched.get_last_lr()[0],
            }
            if log_cb: log_cb(log)
            if self._mlflow_ctx:
                mlflow.log_metrics({
                    "loss": loss.item(),
                    "loss_pol": loss_pol.item(),
                    "loss_val": loss_val.item(),
                    "lr": log["lr"],
                }, step=self._global_step)

            # eval & ckpt every 20 %
            if (ep+1) % max(1, epochs//5) == 0:
                sr = self._quick_eval()
                if self._mlflow_ctx:
                    mlflow.log_metric("success_rate", sr, step=self._global_step)
                self._save_ckpt(f"e{ep+1:02d}")

        if self._mlflow_ctx:
            mlflow.log_metric("train_time_sec", time.time()-start)
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


import os
import pickle
from pathlib import Path

import torch
import numpy as np


# -------- הגדרות כלליות -------- #
AGENT_PATH = Path("peg_agent.pt")
HISTORY_PATH = Path("episode_history.pkl")
TRAIN_EPISODES = 800
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
    return Agent(env, model, asp, ReplayBuffer(), sims=ckpt.get("sims", 100), device=DEVICE, keep_history=True)


# -------- אימון / הרצה -------- #
def train_new_agent(episodes: int = TRAIN_EPISODES) -> Agent:
    env = PegSolitaireEnv(Board, Game)
    asp = PegSolitaireActionSpace(env.board_mask)
    model = PegSolitaireNet(len(asp), device=DEVICE)
    buffer = ReplayBuffer()
    agent = Agent(env, model, asp, buffer, sims=100, device=DEVICE, keep_history=True)

    print("🚀 Starting training loop ...")
    for ep in range(1, episodes + 1):
        agent.self_play_episode()

        if ep % 10 == 0 and len(buffer) >= 1024:
            agent.train(batch=256, epochs=3, lr=1e-3)

        if ep % 100 == 0:
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


# -------- הפעלה ראשית -------- #
if __name__ == "__main__":
    agent = load_agent() if AGENT_PATH.exists() else train_new_agent()

import os
import pickle
import tkinter as tk
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Paths & Config
AGENT_PATH = Path("peg_agent.pt")
HISTORY_PATH = Path("episode_history.pkl")
PLOTS_DIR = Path("plots")
TRAIN_EPISODES = 800
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🧪 Using device: {DEVICE}")

# ------------------ Save / Load ------------------ #
def save_agent(agent: Agent, path: Path = AGENT_PATH):
    data = {
        "state_dict": agent.model.state_dict(),
        "n_actions": len(agent.action_space),
        "sims": agent.mcts.sims,
    }
    torch.save(data, path)
    print(f"✅ Agent saved to: {path.resolve()}")

def load_agent(path: Path = AGENT_PATH) -> Agent:
    if not path.exists():
        raise FileNotFoundError(f"No agent checkpoint at {path}")
    ckpt = torch.load(path, map_location=DEVICE)
    env = PegSolitaireEnv(Board, Game)
    asp = PegSolitaireActionSpace(env.board_mask)
    model = PegSolitaireNet(ckpt["n_actions"], device=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    agent = Agent(env, model, asp, ReplayBuffer(), sims=ckpt.get("sims",100), device=DEVICE, keep_history=True)
    print("📦 Agent loaded.")
    return agent

# ------------------ Training ------------------ #
def train_new_agent(episodes: int = TRAIN_EPISODES) -> Agent:
    env = PegSolitaireEnv(Board, Game)
    asp = PegSolitaireActionSpace(env.board_mask)
    model = PegSolitaireNet(len(asp), device=DEVICE)
    buf = ReplayBuffer()
    agent = Agent(env, model, asp, buf, sims=100, device=DEVICE, keep_history=True)

    print("🚀 Starting training...")
    for ep in range(1, episodes + 1):
        agent.self_play_episode()
        if ep % 10 == 0 and len(buf) >= 1024:
            agent.train(batch=256, epochs=3, lr=1e-3)
        if ep % 100 == 0:
            s = agent.stats
            print(f"  ↳ ep {ep}/{episodes} | buf={len(buf):5d} | succ={s['success']}/{s['episodes']} | avg_moves={s['avg_moves']:.1f}")

    save_agent(agent)
    if agent.episodes:
        pickle.dump(agent.episodes, open(HISTORY_PATH, "wb"))
        print(f"📘 History saved to {HISTORY_PATH.resolve()}")
    return agent

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

# ------------------ GUI ------------------ #
class PegSolitaireGUI(tk.Frame):
    CELL, R, PAD = 60, 22, 16
    PEG, HOLE, OUTL, HILITE = "#FFD600","#202020","#333","#42A5F5"
    SUGGEST, BG = "#00C853", "#eeeeee"
    BAR_W, BAR_H = 160, 16

    def __init__(self, master, game: Game, agent: Agent = None):
        super().__init__(master, bg=self.BG)
        self.game, self.agent = game, agent
        self.sel = None
        self.hint = None

        side = 7*self.CELL + 2*self.PAD
        self.canvas = tk.Canvas(self, width=side, height=side, bg=self.BG, highlightthickness=0)
        self.canvas.pack()
        top = tk.Frame(self,bg=self.BG); top.pack(pady=4, fill="x")
        self.status = tk.Label(top, font=("Arial",14),bg=self.BG,anchor="w")
        self.status.pack(side="left",expand=True,fill="x")
        self.bar = tk.Canvas(top,width=self.BAR_W,height=self.BAR_H,bg=self.BG,highlightthickness=0)
        self.bar.pack(side="right",padx=6)

        btns=tk.Frame(self,bg=self.BG); btns.pack()
        for txt,cmd in [("↩️ Undo",self.on_undo),("↪️ Redo",self.on_redo),("🔄 Reset",self.on_reset),("🤖 Hint",self.on_hint)]:
            tk.Button(btns,text=txt,command=cmd).pack(side="left",padx=3)

        self.log = tk.Listbox(self,width=40,height=6,font=("Consolas",11))
        self.log.pack(pady=(8,0))
        self.canvas.bind("<Button-1>",self.on_click)
        master.bind("<Control-z>", lambda e: self.on_undo())
        master.bind("<Control-y>", lambda e: self.on_redo())
        self.redraw()

    def _xy(self,pos): return (self.PAD+pos[1]*self.CELL+self.CELL//2, self.PAD+pos[0]*self.CELL+self.CELL//2)

    def redraw(self):
        self.canvas.delete("all")
        for pos in Board.LEGAL_POSITIONS:
            x,y=self._xy(pos)
            fill=self.PEG if self.game.board.get(pos)==1 else self.HOLE
            width = 3 if pos==self.sel else 1
            outline=self.HILITE if pos==self.sel else self.OUTL
            self.canvas.create_oval(x-self.R,y-self.R,x+self.R,y+self.R,fill=fill,outline=outline,width=width)

        if self.sel:
            for s,d,_ in self.game.get_legal_moves():
                if s==self.sel:
                    x,y=self._xy(d)
                    self.canvas.create_oval(x-self.R//2,y-self.R//2,x+self.R//2,y+self.R//2,outline=self.HILITE,width=3)

        if self.hint:
            src,dst=self.hint
            x1,y1=self._xy(src); x2,y2=self._xy(dst)
            self.canvas.create_line(x1,y1,x2,y2,fill=self.SUGGEST,width=5,arrow=tk.LAST)

        self._update_status(); self._update_buttons(); self._update_log(); self._update_bar()

    def _update_bar(self):
        v = 0.0
        if self.agent:
            obs=self.game.board.encode_observation()
            t=torch.tensor(obs,dtype=torch.float32,device=self.agent.device).permute(2,0,1).unsqueeze(0)
            with torch.no_grad(): _,v_out=self.agent.model(t)
            v=float(v_out)
        frac=(v+1)/2
        length=int(frac*self.BAR_W)
        col="#d50000" if v<-0.3 else "#9e9e9e" if v<0.3 else "#00c853"
        self.bar.delete("all")
        self.bar.create_rectangle(0,0,length,self.BAR_H,fill=col,width=0)
        self.bar.create_rectangle(0,0,self.BAR_W,self.BAR_H,outline="#555")

    def on_click(self,e):
        pos=((e.y-self.PAD)//self.CELL,(e.x-self.PAD)//self.CELL)
        if pos not in Board.LEGAL_POSITIONS: return
        if self.sel is None and self.game.board.get(pos)==1: self.sel=pos
        elif self.sel and pos!=self.sel:
            success,_=self.game.apply_move(self.sel,pos)[:2]
            if success:
                self.sel=self.hint=None
        else: self.sel=None
        self.redraw()

    def on_undo(self): self._call(self.game.undo)
    def on_redo(self): self._call(self.game.redo)
    def on_reset(self): self._call(self.game.reset)

    def _call(self,fn):
        if not fn(): return
        self.sel=self.hint=None
        self.redraw()

    def on_hint(self):
        if not self.agent:
            tk.messagebox.showinfo("Hint","No agent loaded.")
            return
        if self.game.is_game_over():
            self.status.config(text="Game over.")
            return
        obs=self.game.board.encode_observation()
        π = self.agent.mcts.run(obs, tau=0.0)
        legal=self.game.get_legal_moves()
        mask=self.agent.action_space.legal_action_mask(legal)
        idx=int(np.argmax(π*mask))
        action=self.agent.action_space.from_index(idx)
        dr,dc=Game.DIRECTIONS[action[2]]
        self.hint=((action[0],action[1]),(action[0]+dr,action[1]+dc))
        self.redraw()

    def _update_status(self):
        if self.game.is_win():
            text="Victory! Single peg in center."
        elif self.game.is_game_over():
            text=f"Game Over in {len(self.game.move_log)} moves."
        else:
            text=f"Pegs: {self.game.board.count_pegs()} | Moves: {len(self.game.move_log)}"
        self.status.config(text=text)

    def _update_buttons(self):
        pass  # disabled for brevity

    def _update_log(self):
        self.log.delete(0,tk.END)
        for i, (s,_,d) in enumerate(self.game.move_log,1):
            self.log.insert(tk.END, f"{i:2}: {s}→{d}")

# Main
if __name__ == "__main__":
    agent = load_agent() if AGENT_PATH.exists() else train_new_agent()
    generate_plots()
    root = tk.Tk()
    root.title("Peg-Solitaire AI")
    PegSolitaireGUI(root, Game(), agent).pack()
    root.mainloop()