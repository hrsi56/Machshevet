from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Dict

import numpy as np

Pos = Tuple[int, int]        # (row, col)

# ------------------------------------------------------------
#  קבועים גלובליים: גאומטריה ומיפוי ביטים
# ------------------------------------------------------------
BOARD_SIZE      = 7
LEGAL_POSITIONS: Tuple[Pos, ...] = tuple(
    (r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
    if (2 <= r <= 4) or (2 <= c <= 4)
)
BIT_IDX: Dict[Pos, int] = {p: i for i, p in enumerate(LEGAL_POSITIONS)}  # Pos → 0..32
TOTAL_PEGS      = len(LEGAL_POSITIONS) - 1         # 32 (מרכז ריק)
CENTER_BIT      = 1 << BIT_IDX[(3, 3)]

LEGAL_MASK = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
for r, c in LEGAL_POSITIONS:
    LEGAL_MASK[r, c] = 1.0
OBS_MASK = LEGAL_MASK[..., None]                   # לערוץ-מסכה בתצפית

# ------------------------------------------------------------
#  מחלקת הלוח – גרסת Bitboard
# ------------------------------------------------------------
@dataclass(slots=True)
class Board:
    bits: int = field(default=(1 << (TOTAL_PEGS + 1)) - 1 ^ CENTER_BIT)  # 32 פגים, חור מרכזי
    peg_count: int = field(default=TOTAL_PEGS)                           # נשמר מעודכן

    # ---------- בסיס ----------
    def reset(self) -> None:
        self.bits = (1 << (TOTAL_PEGS + 1)) - 1 ^ CENTER_BIT
        self.peg_count = TOTAL_PEGS

    def get(self, pos: Pos) -> int:
        return (self.bits >> BIT_IDX[pos]) & 1

    def set(self, pos: Pos, val: int) -> None:
        idx = BIT_IDX[pos]
        mask = 1 << idx
        if val:
            if not self.bits & mask:
                self.bits |= mask
                self.peg_count += 1
        else:
            if self.bits & mask:
                self.bits &= ~mask
                self.peg_count -= 1

    # ---------- איסוף ----------
    def pegs(self) -> List[Pos]:
        """Locations of all pegs (cost O(n_pegs))."""
        b = self.bits
        return [p for p in LEGAL_POSITIONS if (b >> BIT_IDX[p]) & 1]

    def holes(self) -> List[Pos]:
        b = self.bits
        return [p for p in LEGAL_POSITIONS if not (b >> BIT_IDX[p]) & 1]

    def count_pegs(self) -> int:
        return self.peg_count

    # ---------- array / dict ----------
    def to_array(self) -> np.ndarray:
        """Return (7,7) float array with 0/1 only on legal squares."""
        # פריסה מהירה של 49-bit ל-(7,7)
        bits64 = np.uint64(self.bits).view(np.uint8)
        flat = np.unpackbits(bits64)[-len(LEGAL_POSITIONS):]  # array([b32 … b0])
        arr = np.zeros((BOARD_SIZE, BOARD_SIZE), np.float32)
        for val, (r, c) in zip(flat[::-1], LEGAL_POSITIONS):
            arr[r, c] = float(val)
        return arr

    # alias לשם תאימות
    as_array = to_array
    get_state = to_array

    def set_state(self, data: Union[np.ndarray, Dict[Pos, int]]) -> None:
        """Load board from dict or (7,7) array."""
        if isinstance(data, dict):
            self.bits = 0
            for p, v in data.items():
                if v:
                    self.bits |= (1 << BIT_IDX[p])
            self.peg_count = bin(self.bits).count("1")
        else:
            arr = np.asarray(data, dtype=np.float32)
            if arr.shape != (BOARD_SIZE, BOARD_SIZE):
                raise ValueError("state array must be 7×7")
            self.bits = 0
            for p in LEGAL_POSITIONS:
                if arr[p] == 1:
                    self.bits |= (1 << BIT_IDX[p])
            self.peg_count = bin(self.bits).count("1")

    def to_dict(self) -> Dict[Pos, int]:
        b = self.bits
        return {p: int((b >> BIT_IDX[p]) & 1) for p in LEGAL_POSITIONS}

    # ---------- NN-encoding ----------
    def encode_observation(self) -> np.ndarray:
        peg_cnt = float(self.peg_count)
        removed = (TOTAL_PEGS - peg_cnt) / TOTAL_PEGS
        remaining = peg_cnt / TOTAL_PEGS

        obs = np.zeros((BOARD_SIZE, BOARD_SIZE, 4), dtype=np.float32)
        obs[..., 0] = self.to_array()
        obs[..., 1].fill(removed)
        obs[..., 2].fill(remaining)
        obs[..., 3] = LEGAL_MASK
        return obs

    # ---------- אוגמנטציה ----------
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

    # ---------- שירות ----------
    def copy(self) -> "Board":
        return Board(bits=self.bits, peg_count=self.peg_count)

    clone = copy

    def __hash__(self) -> int:
        return hash(self.bits)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Board) and self.bits == other.bits

    def __str__(self) -> str:
        arr = self.to_array()
        rows = [
            " ".join("●" if arr[r, c] == 1 else "◯" if arr[r, c] == 0 else " "
                     for c in range(BOARD_SIZE))
            for r in range(BOARD_SIZE)
        ]
        return "\n".join(rows)


from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple

import numpy as np

# ------------------------------------------------------------
#  סוגי עזר וקבועים שהוגדרו במחלקת Board
# ------------------------------------------------------------
Pos = Tuple[int, int]                       # (row, col)
Action = Tuple[int, int, int]               # (row, col, dir-idx)


# 4 כיוונים – תואם לאינדקסים שהשתמשת עד כה
class Dir(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

DIR_VECTORS: Tuple[Pos, ...] = ((-2, 0), (2, 0), (0, -2), (0, 2))

# ------------------------------------------------------------
#  טבלאות שכנות מוכנות – מעבירות מהר מ-idx ל-idx
# ------------------------------------------------------------
_NUM_POS = len(LEGAL_POSITIONS)             # 33
_NEIGH_OV   = np.full((_NUM_POS, 4), -1, np.int8)   # src_idx × dir → over_idx
_NEIGH_DST  = np.full((_NUM_POS, 4), -1, np.int8)   # src_idx × dir → dst_idx

for idx, (r, c) in enumerate(LEGAL_POSITIONS):
    for d, (dr, dc) in enumerate(DIR_VECTORS):
        over = (r + dr // 2, c + dc // 2)
        dst  = (r + dr,       c + dc)
        if over in BIT_IDX and dst in BIT_IDX:
            _NEIGH_OV[idx, d]  = BIT_IDX[over]
            _NEIGH_DST[idx, d] = BIT_IDX[dst]

# ------------------------------------------------------------
#  תיעוד מהלך קצר – חוסך זיכרון
# ------------------------------------------------------------
@dataclass(slots=True)
class MoveRec:
    src:  int
    over: int
    dst:  int
    prev_bits: int
    prev_count: int

# ------------------------------------------------------------
#  Game – מנוע פג-סוליטר מהיר
# ------------------------------------------------------------
class Game:
    """ Peg-Solitaire engine, מותאם ל-Bitboard ו-PBRS. """

    def __init__(
        self,
        board: Optional[Board] = None,
        reward_fn: Optional[Callable[[bool, float, float], float]] = None,
        potential_fn: Optional[Callable[[Board], float]] = None,
    ) -> None:
        self.board: Board = board.copy() if board else Board()
        self._potential_fn = potential_fn or (lambda b: -b.count_pegs())
        self._reward_fn = reward_fn or self._default_reward

        self.history: List[MoveRec] = []
        self.redo_stack: List[MoveRec] = []
        self.last_move: Optional[Tuple[Pos, Pos, Pos]] = None
        self.meta: Dict[str, object] = {}

    # ---------- לוגיקת מהלכים מהירה ----------
    def is_legal_move_idx(self, src_i: int, d: int) -> Tuple[bool, int, int]:
        """Return (ok, over_idx, dst_idx)."""
        over_i = _NEIGH_OV[src_i, d]
        dst_i  = _NEIGH_DST[src_i, d]
        if over_i == -1:
            return False, -1, -1
        bits = self.board.bits
        if (bits >> over_i) & 1 and not (bits >> dst_i) & 1:
            return True, over_i, dst_i
        return False, -1, -1

    def get_legal_moves(self) -> List[Tuple[Pos, Pos, Pos]]:
        moves = []
        bits = self.board.bits
        for src_i in range(_NUM_POS):
            if not (bits >> src_i) & 1:
                continue
            for d in range(4):
                ok, over_i, dst_i = self.is_legal_move_idx(src_i, d)
                if ok:
                    moves.append((
                        LEGAL_POSITIONS[src_i],
                        LEGAL_POSITIONS[dst_i],
                        LEGAL_POSITIONS[over_i],
                    ))
        return moves

    def get_legal_actions(self) -> List[Action]:
        acts: List[Action] = []
        bits = self.board.bits
        for src_i, (r, c) in enumerate(LEGAL_POSITIONS):
            if not (bits >> src_i) & 1:
                continue
            for d in range(4):
                ok, _, dst_i = self.is_legal_move_idx(src_i, d)
                if ok:
                    acts.append((r, c, d))
        return acts

    def is_legal_action(self, a: Action) -> bool:
        r, c, d = a
        src_i = BIT_IDX.get((r, c), -1)
        if src_i == -1 or ((self.board.bits >> src_i) & 1) == 0:
            return False
        return self.is_legal_move_idx(src_i, d)[0]

    # ---------- יישום מהלך ----------
    def _apply_idx(self, src_i: int, over_i: int, dst_i: int) -> None:
        bits = self.board.bits
        # כיבוי src+over, הדלקת dst
        bits &= ~(1 << src_i)
        bits &= ~(1 << over_i)
        bits |=  (1 << dst_i)
        self.board.bits = bits
        self.board.peg_count -= 1  # ירד פיון אחד

    def apply_action(self, a: Action) -> Tuple[bool, float, bool, Dict]:
        r, c, d = a
        src_i = BIT_IDX.get((r, c), -1)
        if src_i == -1:
            return False, -1.0, self.is_game_over(), {"reason": "illegal pos"}
        ok, over_i, dst_i = self.is_legal_move_idx(src_i, d)
        if not ok:
            return False, -1.0, self.is_game_over(), {"reason": "illegal move"}

        # ----- PBRS -----
        phi_before = self._potential_fn(self.board)

        # רשום היסטוריה
        self.history.append(
            MoveRec(src_i, over_i, dst_i, self.board.bits, self.board.peg_count)
        )
        self.redo_stack.clear()

        # בצע
        self._apply_idx(src_i, over_i, dst_i)
        self.last_move = (
            LEGAL_POSITIONS[src_i],
            LEGAL_POSITIONS[over_i],
            LEGAL_POSITIONS[dst_i],
        )

        done = self.is_game_over()
        phi_after = self._potential_fn(self.board)
        reward = self._reward_fn(done, phi_before, phi_after)

        info = {
            "done": done,
            "is_solved": self.is_win(),
            "num_pegs": self.board.peg_count,
            "last_move": self.last_move,
        }
        return True, reward, done, info

    # נוחות: alias
    apply_move = apply_action

    # ---------- Undo / Redo ----------
    def undo(self) -> bool:
        if not self.history:
            return False
        rec = self.history.pop()
        self.redo_stack.append(rec)
        self.board.bits = rec.prev_bits
        self.board.peg_count = rec.prev_count
        self.last_move = None if not self.history else (
            LEGAL_POSITIONS[self.history[-1].src],
            LEGAL_POSITIONS[self.history[-1].over],
            LEGAL_POSITIONS[self.history[-1].dst],
        )
        return True

    def redo(self) -> bool:
        if not self.redo_stack:
            return False
        rec = self.redo_stack.pop()
        self.history.append(rec)
        self._apply_idx(rec.src, rec.over, rec.dst)
        self.last_move = (
            LEGAL_POSITIONS[rec.src],
            LEGAL_POSITIONS[rec.over],
            LEGAL_POSITIONS[rec.dst],
        )
        return True

    # ---------- סטטוס ----------
    def is_game_over(self) -> bool:
        return self.board.peg_count == 1 or not self.get_legal_moves()

    def is_win(self) -> bool:
        return (
            self.board.peg_count == 1
            and (self.board.bits & CENTER_BIT) != 0
        )

    # ---------- Utility ----------
    def reset(self, board: Optional[Board] = None) -> None:
        self.board = board.copy() if board else Board()
        self.history.clear()
        self.redo_stack.clear()
        self.last_move = None
        self.meta.clear()

    def clone_state(self) -> "Game":
        g = Game(
            board=self.board.copy(),
            reward_fn=self._reward_fn,
            potential_fn=self._potential_fn,
        )
        g.history = self.history.copy()
        g.redo_stack = self.redo_stack.copy()
        g.last_move = self.last_move
        return g

    # ---------- ברירת־מחדל PBRS ----------
    @staticmethod
    def _default_reward(done: bool, potential_before: float, potential_after: float) -> float:
        gamma = 0.995
        step_floor = -0.02
        bonus_win = 20.0
        base_penalty = -3.0
        k_pen = 1.0

        delta_phi = gamma * potential_after - potential_before
        reward = delta_phi if delta_phi > 0 else step_floor

        if not done:
            return reward
        if potential_after == 0:          # זכינו – כל הפגים ירדו, מרכז מלא
            return reward + bonus_win
        # כשלון
        # (ה-potential כבר שלילי, אבל נוסיף קנס לפי כמות פגים)
        pegs_left = int(-potential_after) + 1
        return reward + base_penalty - k_pen * (pegs_left - 1)

    # ---------- הדפסה ----------
    def __str__(self) -> str:
        parts = [str(self.board)]
        if self.last_move:
            parts.append(f"Last move: {self.last_move}")
        return "\n".join(parts)




"""Peg-Solitaire RL – FULL STACK
=================================
Drop‑in, single‑file implementation containing:

* **Board** & **Game** – in their own modules (see earlier).  
* **PegSolitaireEnv** – Gym‑lite wrapper.  
* **Networks** – `ValueNetwork`, `PegSolitaireNet`.  
* **Prioritised ReplayBuffer** – sum‑tree implementation.  
* **MCTS** – batched PUCT with root Dirichlet noise.  
* **PegSolitaireActionSpace** – helper for (row,col,dir) → index.  
* **Agent** – AlphaZero‑style self‑play + training loop with optional MLflow.

The file assumes Python ≥ 3.9, `numpy`, `torch`, and (optionally) `mlflow`.
Feel free to split into modules if preferred – everything here is self‑contained.
"""


import heapq, itertools, math, os, random, time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
#  Optional MLflow -------------------------------------------------------------
# -----------------------------------------------------------------------------
try:
    import mlflow  # type: ignore
    _HAVE_MLFLOW = True
except ImportError:  # graceful fallback
    _HAVE_MLFLOW = False

# -----------------------------------------------------------------------------
#  Local project imports (provided earlier) ------------------------------------
# -----------------------------------------------------------------------------

Pos = Tuple[int, int]
Action = Tuple[int, int, int]

# =============================================================================
#  NETWORKS -------------------------------------------------------------------
# =============================================================================
class ValueNetwork(nn.Module):
    def __init__(self, channels: int = 32, hidden: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(4, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels * 2, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels * 2)
        self.pool  = nn.AdaptiveAvgPool2d(1)
        self.fc1   = nn.Linear(channels * 2, hidden)
        self.fc2   = nn.Linear(hidden, 1)
        self.apply(self._init)
    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
    def forward(self, x: torch.Tensor):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(channels)
        self.c2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        skip = x
        x = F.relu(self.b1(self.c1(x)))
        x = self.b2(self.c2(x))
        return F.relu(x + skip)

class PegSolitaireNet(nn.Module):
    def __init__(self, n_actions: int, channels: int = 64, n_blocks: int = 6,
                 hidden_value: int = 128, dropout: float = 0.1,
                 device: str | torch.device = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.stem = nn.Sequential(
            nn.Conv2d(4, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(n_blocks)])
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=4, batch_first=True)
        self.pol_ln = nn.LayerNorm(channels)
        self.pol_fc = nn.Linear(channels, n_actions)
        self.val_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * 7 * 7, hidden_value),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_value, 1),
            nn.Tanh()
        )
        self.apply(self._init)
        self.to(self.device)
    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
    def _policy_head(self, feats: torch.Tensor):
        B, C, _, _ = feats.shape
        tokens = feats.flatten(2).transpose(1, 2)           # (B, 49, C)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        pooled = self.pol_ln(attn_out.mean(dim=1))
        return self.pol_fc(pooled)
    def forward(self, x: torch.Tensor):
        if x.device != self.device:
            x = x.to(self.device, non_blocking=True)
        x = self.body(self.stem(x))
        logits = self._policy_head(x)
        value  = self.val_head(x).squeeze(-1)
        return logits, value

# =============================================================================
#  ENVIRONMENT ----------------------------------------------------------------
# =============================================================================
class PegSolitaireEnv:
    BOARD_SIZE = 7
    TOTAL_PEGS = 32

    def __init__(self, board_cls = Board, game_cls = Game,
                 reward_fn: Optional[Callable[[bool,float,float],float]] = None,
                 potential_fn: Optional[Callable[[Board],float]] = None):
        self.board_mask = board_cls.LEGAL_MASK.astype(np.float32).copy()
        self._phi = potential_fn or (lambda b: float(-b.peg_count + 0.5 * bool(b.bits & CENTER_BIT)))
        self.game = game_cls(board_cls(), reward_fn=reward_fn, potential_fn=self._phi)
        self.done = False
    # ---------------- observation ----------------
    def encode_observation(self):
        arr = self.game.board.to_array()
        peg_cnt = float(self.game.board.peg_count)
        removed = (self.TOTAL_PEGS - peg_cnt) / self.TOTAL_PEGS
        remain  = peg_cnt / self.TOTAL_PEGS
        obs = np.empty((7,7,4), np.float32)
        obs[...,0] = arr
        obs[...,1].fill(removed)
        obs[...,2].fill(remain)
        obs[...,3] = self.board_mask
        return obs
    # ---------------- gym API ---------------
    def reset(self, state:Optional[Union[dict,Board]] = None):
        if state is None:
            self.game.reset()
        else:
            self.game.set_state(state)
        self.done = False
        return self.encode_observation(), {"num_pegs": self.game.board.peg_count}
    def step(self, action:Action):
        if self.done:
            raise RuntimeError("step() after episode finished – reset().")
        if self.game.is_legal_action(action):
            _, r, self.done, _ = self.game.apply_action(action)
        else:
            r = -1.0
            self.done = self.game.is_game_over()
        return self.encode_observation(), r, self.done, {
            "num_pegs": self.game.board.peg_count,
            "is_solved": self.game.is_solved(),
        }
    # helper
    def get_legal_actions(self):
        return self.game.get_legal_actions()
    def clone_state(self, state=None):
        clone = PegSolitaireEnv()
        clone.game = self.game.clone_state()
        return clone

# =============================================================================
#  ACTION SPACE ----------------------------------------------------------------
# =============================================================================
class PegSolitaireActionSpace:
    DIRS = 4  # ↑ ↓ ← →
    def __init__(self, board_mask: np.ndarray):
        self.valid_cells = [(r,c) for r in range(7) for c in range(7) if board_mask[r,c]==1]
        self.actions = [(r,c,d) for r,c in self.valid_cells for d in range(self.DIRS)]
        self._to_idx = {a:i for i,a in enumerate(self.actions)}
    def to_index(self, a:Action):
        return self._to_idx[a]
    def from_index(self, idx:int):
        return self.actions[idx]
    def legal_action_mask(self, legal:List[Action]):
        mask = np.zeros(len(self.actions), np.float32)
        for a in legal:
            mask[self._to_idx[a]] = 1.0
        return mask
    def __len__(self):
        return len(self.actions)

# =============================================================================
#  REPLAY BUFFER (sum‑tree PER) ------------------------------------------------
# =============================================================================
class _SumTree:
    def __init__(self, cap:int):
        self.cap = 1 << (cap-1).bit_length()
        self.tree = np.zeros(2*self.cap, np.float32)
        self.size = 0
    def total(self):
        return float(self.tree[1])
    def _prop(self, idx:int, delta:float):
        while idx>1:
            idx//=2; self.tree[idx]+=delta
    def set(self, idx_data:int, p:float):
        idx = idx_data+self.cap; delta = p-self.tree[idx]
        self.tree[idx]=p; self._prop(idx, delta)
    def get(self, v:float):
        idx=1
        while idx<self.cap:
            left=idx*2
            if v<=self.tree[left]: idx=left
            else: v-=self.tree[left]; idx=left+1
        data_idx=idx-self.cap
        return data_idx, self.tree[idx]

class ReplayBuffer:
    def __init__(self, max_size:int=50_000, alpha:float=0.6, beta:float=0.4):
        self.max_size=max_size; self.alpha=alpha; self.beta=beta
        self.tree=_SumTree(max_size)
        self.data:[Tuple[np.ndarray,np.ndarray,float]]=[None]*max_size
        self.next_idx=0
    def __len__(self):
        return self.tree.size
    def push(self, sample:Tuple[np.ndarray,np.ndarray,float], priority:Optional[float]=None):
        if priority is None:
            priority=self.tree.tree[self.next_idx+self.tree.cap] or 1.0
        self.data[self.next_idx]=sample
        self.tree.set(self.next_idx, priority**self.alpha)
        self.next_idx=(self.next_idx+1)%self.max_size
        self.tree.size=min(self.tree.size+1, self.max_size)
    def sample_as_tensors(self, batch:int, device:str|torch.device="cpu"):
        device=torch.device(device)
        seg=self.tree.total()/batch
        idxs,w=[],[]
        for i in range(batch):
            v=random.uniform(seg*i, seg*(i+1))
            idx,p=self.tree.get(v); idxs.append(idx)
            prob=p/self.tree.total(); w.append((len(self)*prob)**(-self.beta))
        w=np.asarray(w,np.float32); w/=w.max()+1e-8
        obs=torch.tensor(np.stack([self.data[i][0] for i in idxs]),dtype=torch.float32,device=device)\
              .permute(0,3,1,2).contiguous()
        pi=torch.tensor(np.stack([self.data[i][1] for i in idxs]),dtype=torch.float32,device=device)
        gt=torch.tensor([self.data[i][2] for i in idxs],dtype=torch.float32,device=device)
        wt=torch.tensor(w,dtype=torch.float32,device=device)
        return obs,pi,gt,idxs,wt
    def update_priorities(self, idxs:List[int], new_p:np.ndarray):
        for i,p in zip(idxs,new_p):
            self.tree.set(i,float(p)**self.alpha)

# =============================================================================
#  MCTS -----------------------------------------------------------------------
# =============================================================================
class _Node:
    __slots__ = ("prior", "children", "visit", "value_sum")
    def __init__(self, prior: float):
        self.prior = prior
        self.children: Dict[int, "_Node"] = {}
        self.visit = 0
        self.value_sum = 0.0
    @property
    def value(self):
        return self.value_sum / self.visit if self.visit else 0.0

class MCTS:
    def __init__(
        self,
        env: PegSolitaireEnv,
        model: PegSolitaireNet,
        action_space: PegSolitaireActionSpace,
        sims: int = 96,
        c_puct: float = 1.5,
        root_noise: bool = True,
        dir_alpha: float = 0.3,
        noise_ratio: float = 0.25,
        device: str | torch.device = "cpu",
    ) -> None:
        self.env, self.model, self.action_space = env, model, action_space
        self.sims, self.c_puct = sims, c_puct
        self.root_noise, self.dir_alpha, self.noise_ratio = root_noise, dir_alpha, noise_ratio
        self.device = torch.device(device)

    # ------------------ helpers ------------------
    def _make_children(self, probs: np.ndarray, legal: List[Action]):
        mask = self.action_space.legal_action_mask(legal)
        probs *= mask
        probs = probs / mask.sum() if probs.sum() > 1e-8 else mask / mask.sum()
        if self.root_noise:
            legal_idx = np.where(mask == 1.0)[0]
            if len(legal_idx):
                noise = np.random.dirichlet([self.dir_alpha] * len(legal_idx))
                probs[legal_idx] = (
                    (1 - self.noise_ratio) * probs[legal_idx]
                    + self.noise_ratio * noise
                )
        return {
            self.action_space.to_index(a): _Node(float(probs[self.action_space.to_index(a)]))
            for a in legal
        }

    def _expand(self, obs: np.ndarray, legal: List[Action]):
        x = (
            torch.tensor(obs, dtype=torch.float32, device=self.device)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        logits, _ = self.model(x)
        probs = torch.softmax(logits, -1).cpu().numpy().reshape(-1)
        node = _Node(1.0)
        node.children = self._make_children(probs, legal)
        return node

    def _select(self, node: "_Node") -> Tuple[int, "_Node"]:
        best, bidx = -1e9, -1
        sqrt_visits = math.sqrt(node.visit)
        for idx, child in node.children.items():
            u = self.c_puct * child.prior * sqrt_visits / (1 + child.visit)
            score = child.value + u
            if score > best:
                best, bidx = score, idx
        return bidx, node.children[bidx]

    def _backprop(self, path: List["_Node"], v: float):
        for n in path:
            n.visit += 1
            n.value_sum += v
            v = -v  # alternate perspective

    # ------------------ main --------------------
    def run(self, root_obs: np.ndarray, tau: float = 1.0, entropy_stop: float = 0.6):
        root = self._expand(root_obs, self.env.get_legal_actions())
        for sim in range(self.sims):
            env_c = self.env.clone_state()
            node = root
            path = [root]
            done = False
            while node.children and not done:
                a_idx, node = self._select(node)
                _, _, done, _ = env_c.step(self.action_space.from_index(a_idx))
                path.append(node)
            if done:
                v = 1.0 if env_c.game.is_win() else -1.0
                self._backprop(path, v)
                continue
            obs = env_c.encode_observation()
            leaf = self._expand(obs, env_c.get_legal_actions())
            node.children = leaf.children
            x = (
                torch.tensor(obs, dtype=torch.float32, device=self.device)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            _, v = self.model(x)
            self._backprop(path, float(v.item()))

            visits = np.array([c.visit for c in root.children.values()], np.float32)
            p = visits / (visits.sum() + 1e-8)
            ent = -(p * np.log(p + 1e-9)).sum()
            if ent < entropy_stop and sim > self.sims * 0.4:
                break

        child_items = sorted(root.children.items())
        visits = np.array([c.visit for _, c in child_items], np.float32)
        indices = np.array([idx for idx, _ in child_items])
        if tau == 0.0:
            pi = np.zeros_like(visits)
            pi[np.argmax(visits)] = 1.0
        else:
            log_v = np.log(visits + 1e-8) / tau
            pi = np.exp(log_v - log_v.max())
            pi /= pi.sum()
        full_pi = np.zeros(len(self.action_space), np.float32)
        full_pi[indices] = pi
        return full_pi

# =============================================================================
#  ANALYZER -------------------------------------------------------------------
# =============================================================================
class AgentAnalyzer:
    def __init__(self, action_space: PegSolitaireActionSpace):
        self.action_space = action_space
        self.reset()
    def reset(self):
        self.records: List[Dict] = []
    def log(self, obs, act_idx, value_est, legal_idx):
        self.records.append(dict(obs=obs, action_idx=int(act_idx), value=float(value_est), legal=legal_idx))
    def summary(self):
        if not self.records:
            return {"total_moves": 0}
        counts = defaultdict(int)
        for r in self.records:
            counts[r["action_idx"]] += 1
        top = max(counts.items(), key=lambda kv: kv[1])
        return dict(total_moves=len(self.records), unique_actions=len(counts), top_action=self.action_space.from_index(top[0]), top_freq=top[1])

# =============================================================================
#  AGENT ----------------------------------------------------------------------
# =============================================================================
class Agent:
    def __init__(self, env: PegSolitaireEnv, model: PegSolitaireNet, action_space: PegSolitaireActionSpace,
                 buffer: ReplayBuffer, sims:int=64, device:str|torch.device="cpu", keep_history:bool=False,
                 ckpt_dir:str="checkpoints", mlflow_experiment:Optional[str]=None):
        self.device=torch.device(device)
        self.env,self.model,self.action_space,self.buffer=env,model.to(self.device),action_space,buffer
        self.mcts=MCTS(env,model,action_space,sims=sims,device=self.device)
        self.keep_history=keep_history; self.episodes:List[Dict]=[] if keep_history else []
        self.stats=dict(episodes=0,success=0,avg_moves=0.0); self._global_step=0
        self.ckpt_dir=Path(ckpt_dir); self.ckpt_dir.mkdir(exist_ok=True)
        self.analyzer=AgentAnalyzer(action_space)
        self._mlflow_ctx=None
        if _HAVE_MLFLOW and mlflow_experiment:
            mlflow.set_experiment(mlflow_experiment); self._mlflow_ctx=mlflow.start_run()
    # ---------------- policy transform --------------
    def _transform_policy(self,pi:np.ndarray,rot:int,flip:bool):
        if rot==0 and not flip: return pi
        new=np.zeros_like(pi)
        for idx,p in enumerate(pi):
            if p<1e-9: continue
            a=self.action_space.from_index(idx)
            a_aug=self.env.augment_action(a,rot,flip)
            new[self.action_space.to_index(a_aug)]=p
        return new
    # ---------------- self-play ---------------------
    def self_play_episode(self,augment:bool=True,gamma:float=0.995,n_step:int=6,win_bonus:float=5.0,oversample:int=3):
        obs,_=self.env.reset(); done=False; moves=0; S,P,R=[],[],[]
        self.analyzer.reset()
        while not done:
            moves+=1; tau=1.0 if moves<10 else 0.05
            pi=self.mcts.run(obs,tau=tau)
            legal=self.env.get_legal_actions(); legal_idx=[self.action_space.to_index(a) for a in legal]
            mask=np.zeros_like(pi); mask[legal_idx]=1.0; pi*=mask; pi=pi/ (pi.sum()+1e-9)
            a_idx=int(np.random.choice(len(pi),p=pi)); action=self.action_space.from_index(a_idx)
            S.append(obs.copy()); P.append(pi.copy())
            obs,r,done,_=self.env.step(action); R.append(float(r))
            with torch.no_grad():
                v=self.model(torch.tensor(obs,dtype=torch.float32,device=self.device).permute(2,0,1).unsqueeze(0))[1]
            self.analyzer.log(obs,a_idx,v.item(),legal_idx)
        solved=self.env.game.is_win(); final=win_bonus if solved else -win_bonus
        G=np.zeros(len(R),np.float32); acc=0.0
        for t in reversed(range(len(R))):
            acc=R[t]+gamma*acc; h=min(n_step,len(R)-t)
            G[t]=acc+(gamma**n_step)*final if h==n_step else acc
        rep=oversample if solved else 1
        for s,pi,g in zip(S,P,G):
            for _ in range(rep):
                if augment:
                    for rot in range(4):
                        for flip in (False,True):
                            s_aug=np.rot90(s,rot,(0,1)); s_aug=np.flip(s_aug,1) if flip else s_aug
                            self.buffer.push((s_aug,self._transform_policy(pi,rot,flip),g))
                else:
                    self.buffer.push((s,pi,g))
        st=self.stats; st["episodes"]+=1; st["avg_moves"]=(st["avg_moves"]*(st["episodes"]-1)+moves)/st["episodes"]
        if solved: st["success"]+=1
        if self.keep_history:
            self.episodes.append(dict(reward=float(np.sum(R)),solved=solved,moves=moves,analyzer=self.analyzer.summary()))
    # --------------- training -----------------------
    def train(self,batch:int=256,epochs:int=1,lr:float=1e-3):
        if len(self.buffer)<batch: return
        wd=1e-4; opt=torch.optim.AdamW(self.model.parameters(),lr=lr,weight_decay=wd)
        sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=lr,steps_per_epoch=max(1,len(self.buffer)//batch),epochs=epochs)
        for ep in range(epochs):
            obs,pi_t,G_t,idxs,wt=self.buffer.sample_as_tensors(batch,self.device)
            opt.zero_grad()
            logits,v_pred=self.model(obs)
            loss_policy=F.kl_div(F.log_softmax(logits,-1),pi_t,reduction="batchmean")
            td=v_pred-G_t; loss_value=(wt*td.pow(2)).mean()
            loss=loss_policy+loss_value; loss.backward(); nn.utils.clip_grad_norm_(self.model.parameters(),3.0)
            opt.step(); sched.step(); self._global_step+=1
            self.buffer.update_priorities(idxs, td.detach().abs().cpu().numpy()+1e-6)
    # --------------- utility -----------------------
    def save(self,path:Path):
        torch.save(dict(state_dict=self.model.state_dict(), n_actions=len(self.action_space), sims=self.mcts.sims), path)
    def load(cls,path:Path,device):
        ckpt=torch.load(path,map_location=device)
        env=PegSolitaireEnv>(); asp=PegSolitaireActionSpace(env.board_mask); model=PegSolitaireNet(ckpt["n_actions"],device=device)
        model.load_state_dict(ckpt["state_dict"]); model.eval()
        return cls(env,model,asp,ReplayBuffer(),sims=ckpt.get("sims",64),device=device,keep_history=True)

# =============================================================================
#  QUICK TRAIN SCRIPT + GUI ---------------------------------------------------
# =============================================================================
import pickle, matplotlib.pyplot as plt, tkinter as tk

# -------- הגדרות כלליות -------- #
AGENT_PATH = Path("peg_agent.pt")
HISTORY_PATH = Path("episode_history.pkl")
TRAIN_EPISODES = 400
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🔧 Using device: {DEVICE}")

def save_agent(agent: Agent, path: Path = AGENT_PATH) -> None:
    data = {
        "state_dict": agent.model.state_dict(),
        "n_actions": len(agent.action_space),
        "sims": agent.mcts.sims,
    }
    torch.save(data, path)
    print(f"✅ Agent saved to: {path.resolve()}")


def train_new_agent(
    episodes: int = 400,
    warmup:   int = 50,       # מספר אפיזודות “חימום” לפני PER-IS מלא
    batch:    int = 256,
    lr:       float = 1e-3,
    device:   str | torch.device = "cpu",
) -> Agent:
    """
    מאמן Agent חדש מאפס ומחזיר אותו.
    • self-play + MCTS
    • One-Cycle LR
    • שמירת checkpoint ו-history בקבצים GLOBAL (AGENT_PATH, HISTORY_PATH)
    """

    # --- init ----------------------------------------------------------------
    env   = PegSolitaireEnv()
    asp   = PegSolitaireActionSpace(env.board_mask)
    model = PegSolitaireNet(len(asp), device=device)
    buf   = ReplayBuffer()
    agent = Agent(env, model, asp, buf,
                  sims=64, device=device, keep_history=True)

    print(f"🚀 training on {device}, episodes={episodes}")
    header = "  ↳ ep {ep:4}/{tot} | buf={buf:5} | solved={solv}/{eps} | μ-moves={mu:.1f}"
    t0 = time.time()

    # --- main loop -----------------------------------------------------------
    for ep in range(1, episodes + 1):
        agent.self_play_episode(augment=True)          # מייצר דגימות ל-buffer

        # אימון כשה-buffer התחמם מספיק
        if len(buf) >= 1_024 and (ep >= warmup or len(buf) >= 8_192):
            agent.train(batch=batch,
                        epochs=3 if ep < warmup else 1,
                        lr=lr)

        # log קצר כל 10 אפיזודות
        if ep == 1 or ep % 10 == 0:
            s = agent.stats
            print(header.format(ep=ep, tot=episodes,
                                buf=len(buf),
                                solv=s['success'], eps=s['episodes'],
                                mu=s['avg_moves']))




    save_agent(agent)                                     # לקובץ AGENT_PATH
    if agent.keep_history:
        with open(HISTORY_PATH, "wb") as f:
            pickle.dump(agent.episodes, f)
        print(f"🧾 episode history saved → {HISTORY_PATH.resolve()}")

    print(f"⏱️  total training time: {time.time() - t0:.1f}s")
    return agent