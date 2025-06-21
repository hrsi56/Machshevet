import tkinter as tk
from tkinter import messagebox
import numpy as np
import torch
from pathlib import Path

# ——— ייבוא מהפרויקט שלך ———
from Machshevet import Game, Board, Agent, load_agent   # ודא שהשמות תואמים
# --------------------------------------------------------


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
        self.sel: tuple[int, int] | None = None   # חור נבחר
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
        btns = tk.Frame(self, bg=self.BG);  btns.pack()
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
            x1, y1 = self._xy(src);  x2, y2 = self._xy(dst)
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
    def on_undo(self): self._call(self.game.undo)
    def on_redo(self): self._call(self.game.redo)
    def on_reset(self): self._call(self.game.reset)

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
    try:
        agent = load_agent()
    except FileNotFoundError:
        agent = None
        print("⚠️  Agent checkpoint not found — GUI will run without hints.")

    root = tk.Tk()
    root.title("Peg-Solitaire AI")
    PegSolitaireGUI(root, Game(), agent).pack()
    root.mainloop()