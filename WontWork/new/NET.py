import numpy as np
from scipy.ndimage import convolve
import Game

# (row, col, dir-idx)
import torch
import torch.nn as nn
import torch.nn.functional as F
# ------------------------------------------------------------------
# 🗺️  STRATEGIC MAPS  (7×7 English board – ערכים ניתנים לכיול)
# ------------------------------------------------------------------
from typing import Union, Optional, Callable, Tuple, List

# ================================================================
#  GLOBAL STRATEGIC MAPS  – tuned for maximal solving accuracy
# ================================================================
Pos    = Tuple[int, int]           # (row, col)
Action = Tuple[int, int, int]


CENTRALITY_WEIGHTS = np.array([
    [0.0, 0.0, 0.10, 0.10, 0.10, 0.0, 0.0],
    [0.0, 0.10, 0.20, 0.30, 0.20, 0.10, 0.0],
    [0.10,0.20, 0.40, 0.50, 0.40, 0.20, 0.10],
    [0.10,0.30, 0.50, 1.00, 0.50, 0.30, 0.10],
    [0.10,0.20, 0.40, 0.50, 0.40, 0.20, 0.10],
    [0.0, 0.10, 0.20, 0.30, 0.20, 0.10, 0.0],
    [0.0, 0.0, 0.10, 0.10, 0.10, 0.0, 0.0]], dtype=np.float32)

PAGODA_VALUES = np.array([
    [0,0,1,2,1,0,0],
    [0,1,2,3,2,1,0],
    [1,2,3,4,3,2,1],
    [2,3,4,5,4,3,2],
    [1,2,3,4,3,2,1],
    [0,1,2,3,2,1,0],
    [0,0,1,2,1,0,0]], dtype=np.float32)

DIRS_JUMP = np.array([[-2,0],[2,0],[0,-2],[0,2]], dtype=np.int8)

CORNER_POSITIONS = [(0,3), (3,0), (3,6), (6,3)]
EDGE_MASK = ((CENTRALITY_WEIGHTS < 0.25) & (CENTRALITY_WEIGHTS > 0)).astype(np.float32)  # “true” edges



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
  # אם לא זמין, ראה הערה מתחת


    # משקלות קבועות – שוכנות פעם אחת במחלקה
    _K_REACH = np.array([[0, 1, 0],
                         [1, 0, 1],
                         [0, 1, 0]], dtype=np.int8)  # סופרת שכני “צעד-חצי”

    def _calculate_potential(self, board) -> float:
        arr = board.as_array().astype(np.float32)  # ← שים לב: שימוש ב־board מהפרמטר
        mask = self.board_mask
        pegs = arr * mask
        N = pegs.sum()

        # ---------- φ0 : peg count ----------
        phi0 = -N

        # ---------- φ1 : centrality ----------
        phi1 = (pegs * CENTRALITY_WEIGHTS).sum() / (N + 1e-5)

        # ---------- φ2 : pagoda ----------
        phi2 = (pegs * PAGODA_VALUES).sum()

        # ---------- φ3 : isolation / corner / edge ----------
        if N <= 1:
            phi3 = 0.0
        else:
            neigh = convolve(pegs, self._K_REACH, mode="constant", cval=0)
            reachable = ((neigh > 0) &
                         (convolve(mask - pegs, self._K_REACH, mode="constant", cval=0) > 0))
            isolated = int(((pegs == 1) & (~reachable)).sum())

            corners = int(sum(pegs[r, c] for r, c in CORNER_POSITIONS))
            edges = int((pegs * EDGE_MASK).sum())
            phi3 = -(5.0 * isolated + 2.0 * corners + 1.0 * edges)

        # ---------- φ4 : truly isolated ----------
        truly_isolated = Game._count_truly_isolated_pegs(board)
        phi4 = -10.0 * truly_isolated  # ערך גולמי; ינורמל בהמשך

        # ---------- Z-score normalization ----------
        μ = np.array([-17.0, 0.25, 28.0, -12.0, 2], dtype=np.float32)
        σ = np.array([10.0, 0.12, 20.0, 8.0, 1], dtype=np.float32)
        φ = np.array([phi0, phi1, phi2, phi3, phi4], dtype=np.float32)
        φn = (φ - μ) / (σ + 1e-5)

        w = np.array([0.9, 0.9, 0.6, 0.8, 1], dtype=np.float32)
        total_phi = float((w * φn).sum())

        if getattr(self, "debug_potential", False):
            print(f"[Φ] raw={φ}, norm={φn.round(2)}, Φ={total_phi:.2f}")

        return total_phi

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




import heapq, itertools
import numpy as np
import torch

class ReplayBuffer:
    """
    PER  (α, β)  עם מחיקה לפי priority.
    Stores (obs, π, G_t)
    """

    # ───────────────────────── ctor ─────────────────────────
    def __init__(self,
                 max_size: int = 50_000,
                 alpha   : float = 0.6,
                 beta    : float = 0.4):
        self.max_size = max_size
        self.alpha    = alpha         # כמה “חד” סדר־העדיפויות
        self.beta     = beta          # לאחז IS-weights
        self._cnt     = itertools.count()
        self._heap: List[
            Tuple[float, int, Tuple[np.ndarray, np.ndarray, float]]
        ] = []                        # (priority, order, sample)

    # ───────────────────────── helpers ──────────────────────
    def __len__(self):  return len(self._heap)

    @property           # alias נוח – train() יכול להשתמש ב-_heap או heap
    def heap(self):     return self._heap

    def _priorities(self) -> np.ndarray:
        return np.asarray([abs(e[0]) for e in self._heap], np.float32)

    def _probabilities(self) -> np.ndarray:
        p = self._priorities() ** self.alpha
        return p / (p.sum() + 1e-8)

    # ───────────────────────── push ─────────────────────────
    def push(self,
             sample  : Tuple[np.ndarray, np.ndarray, float],
             priority: Optional[float] = None) -> None:

        if priority is None:
            priority = max((abs(e[0]) for e in self._heap), default=1.0)

        heapq.heappush(self._heap, (float(priority), next(self._cnt), sample))
        if len(self._heap) > self.max_size:
            heapq.heappop(self._heap)

    # ──────────────────────── sampling ──────────────────────
    def sample_as_tensors(
        self,
        batch_size: int,
        device    : str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:

        if not self._heap:
            raise RuntimeError("ReplayBuffer is empty")

        probs = self._probabilities()
        idx   = np.random.choice(len(self._heap),
                                 size=min(batch_size, len(self._heap)),
                                 p=probs)

        # unpack
        obs = torch.tensor(
            np.stack([self._heap[i][2][0] for i in idx]),
            dtype=torch.float32, device=device
        ).permute(0,3,1,2).contiguous()

        pi  = torch.tensor(
            np.stack([self._heap[i][2][1] for i in idx]),
            dtype=torch.float32, device=device)

        G_t = torch.tensor(
            np.asarray([self._heap[i][2][2] for i in idx], np.float32),
            dtype=torch.float32, device=device)

        return obs, pi, G_t, idx

    # ───────────────────── update priorities ─────────────────
    def update_priorities(self,
                          indices      : np.ndarray,
                          new_priorities: np.ndarray) -> None:
        # עדכון במקום, ואז heapify
        for i, p in zip(indices, new_priorities):
            _, order, sample = self._heap[i]
            self._heap[i] = (float(p), order, sample)

        heapq.heapify(self._heap)
