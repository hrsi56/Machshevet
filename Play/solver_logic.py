import pickle
import os
import numpy as np
from numba import njit, int64

# ==========================================
#  חלק 0: האצת Numba (ללא שינוי)
# ==========================================
@njit(int64(int64, int64[:, :, :]))
def fast_canonical_lookup(board, table):
    min_val = board
    c0 = board & 0x7FF
    c1 = (board >> 11) & 0x7FF
    c2 = (board >> 22) & 0x7FF

    for sym_idx in range(8):
        mapped_board = (table[sym_idx, 0, c0] |
                        table[sym_idx, 1, c1] |
                        table[sym_idx, 2, c2])
        if mapped_board < min_val:
            min_val = mapped_board
    return min_val

# ==========================================
#  חלק 1: המנוע (Solver Class)
# ==========================================
class PegSolitaireSolver:
    MEMORY_FILE = "solitaire_pro_brain.pkl"

    def __init__(self):
        self.r_c_to_bit = {}
        self.bit_to_r_c = {}
        self.valid_mask = 0
        self.center_bit = 0

        # מיפוי הלוח
        idx = 0
        for r in range(7):
            for c in range(7):
                if not ((r < 2 and c < 2) or (r < 2 and c > 4) or
                        (r > 4 and c < 2) or (r > 4 and c > 4)):
                    self.r_c_to_bit[(r, c)] = idx
                    self.bit_to_r_c[idx] = (r, c)
                    self.valid_mask |= (1 << idx)
                    if r == 3 and c == 3:
                        self.center_bit = idx
                    idx += 1

        self.symmetry_maps = self._generate_base_symmetry_maps()
        self.lookup_table = self._build_numpy_lookup_table()

        # הכנת מהלכים
        self.moves = []
        directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
        for r in range(7):
            for c in range(7):
                if (r, c) not in self.r_c_to_bit: continue
                src = self.r_c_to_bit[(r, c)]
                for dr, dc in directions:
                    mr, mc = r + dr // 2, c + dc // 2
                    drr, dcc = r + dr, c + dc
                    if (mr, mc) in self.r_c_to_bit and (drr, dcc) in self.r_c_to_bit:
                        mid = self.r_c_to_bit[(mr, mc)]
                        dst = self.r_c_to_bit[(drr, dcc)]
                        mask = (1 << src) | (1 << mid) | (1 << dst)
                        self.moves.append({
                            'mask': mask,
                            'check_src': (1 << src) | (1 << mid),
                            'check_dst': (1 << dst),
                            'src': src, 'dst': dst,
                            'move_key': f"{r},{c}->{drr},{dcc}" # מזהה ייחודי ל-Web
                        })

        self.winning_states = set()
        self.brain_loaded = self.load_memory()

    def _generate_base_symmetry_maps(self):
        maps = []
        for i in range(8):
            mapping = {}
            for r in range(7):
                for c in range(7):
                    if (r, c) not in self.r_c_to_bit: continue
                    rr, cc = r, c
                    if i & 4: rr, cc = cc, rr
                    if i & 1: rr, cc = cc, 6 - rr
                    if i & 2: rr, cc = 6 - rr, 6 - cc
                    src_idx = self.r_c_to_bit[(r, c)]
                    dst_idx = self.r_c_to_bit[(rr, cc)]
                    mapping[src_idx] = dst_idx
            maps.append(mapping)
        return maps

    def _build_numpy_lookup_table(self):
        table = np.zeros((8, 3, 2048), dtype=np.int64)
        for sym_idx in range(8):
            mapping = self.symmetry_maps[sym_idx]
            for chunk_id in range(3):
                bit_offset = chunk_id * 11
                for val in range(2048):
                    transformed_val = 0
                    t = val
                    while t:
                        lsb = t & -t
                        local_idx = lsb.bit_length() - 1
                        real_idx = local_idx + bit_offset
                        if real_idx in mapping:
                            target_idx = mapping[real_idx]
                            transformed_val |= (1 << target_idx)
                        t ^= lsb
                    table[sym_idx, chunk_id, val] = transformed_val
        return table

    def get_canonical(self, board):
        return fast_canonical_lookup(board, self.lookup_table)

    def get_initial_board(self):
        board = self.valid_mask
        board &= ~(1 << self.center_bit)
        return board

    def load_memory(self):
        if os.path.exists(self.MEMORY_FILE):
            try:
                with open(self.MEMORY_FILE, "rb") as f:
                    self.winning_states = pickle.load(f)
                return True
            except Exception:
                return False
        return False

    def get_winning_moves_count(self, board):
        if not self.brain_loaded: return -1 # אינדיקציה שאין מוח
        if self.get_canonical(board) not in self.winning_states and board != (1 << self.center_bit):
            return 0

        count = 0
        for m in self.moves:
            if (board & m['check_src'] == m['check_src']) and (board & m['check_dst'] == 0):
                next_board = board ^ m['mask']
                if self.get_canonical(next_board) in self.winning_states or next_board == (1 << self.center_bit):
                    count += 1
        return count

    def solve_full_path(self, start_board):
        path = []
        if self._find_path_forward(start_board, path):
            return [m['move_key'] for m in path] # מחזיר רשימה של מפתחות מהלכים
        return None

    def _find_path_forward(self, board, path):
        if board == (1 << self.center_bit): return True
        if self.brain_loaded and self.get_canonical(board) not in self.winning_states: return False

        candidates = []
        for m in self.moves:
            if (board & m['check_src'] == m['check_src']) and (board & m['check_dst'] == 0):
                next_board = board ^ m['mask']
                # אם יש מוח, משתמשים בו לסינון, אחרת מנסים הכל (DFS פשוט - איטי מאוד ללא מוח)
                if not self.brain_loaded or self.get_canonical(next_board) in self.winning_states:
                    candidates.append((m, next_board))

        # יוריסטיקה: מיון לפי מספר המהלכים המנצחים בהמשך
        if self.brain_loaded:
            candidates.sort(key=lambda x: self.get_winning_moves_count(x[1]), reverse=True)

        for m, next_b in candidates:
            path.append(m)
            if self._find_path_forward(next_b, path):
                return True
            path.pop()
        return False