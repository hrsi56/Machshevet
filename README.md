# Machshevet (Peg Solitaire) Solver & Analytics 

> **A high-performance Hybrid-BFS Oracle comprising ~1.6 million unique winning states.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Acceleration](https://img.shields.io/badge/Acceleration-Numba_JIT-orange)
![State Space](https://img.shields.io/badge/Winning_States-1,679,072-success)

Most Peg Solitaire solvers are simple recursive backtracking scripts that find *one* solution. **This is not that.**

This project is a mathematical "Oracle" for the Standard English Board (33 holes). It doesn't just "solve" the game; it maps the entire universe of valid gameplay. It knows exactly how many winning paths exist from your current state, and it knows the moment you've made a fatal mistake—often 15 moves before the game actually ends.

you can play it on https://hrsi56.pythonanywhere.com

---

##  The Story of Failure (And How We Fixed It)

This wasn't a straight line to a solution. It was a series of humbling failures.

### Phase 1: The AI Overkill (Failure) ❌
Like many engineers, my first instinct was "let's train a model." We implemented **AlphaZero** with MCTS and reward shaping using Pagoda functions.
* **The Reality:** It was complete overkill. We aren't playing Go; we are playing on a tiny 33-bit grid.
* **The Dealbreaker:** The reward signal is painfully sparse. You only "win" on the very last move. Despite complex reward shaping, the neural network struggled to converge on a perfect solution for a deterministic puzzle.

### Phase 2: Reverse BFS & The "Garden of Eden" (Failure) ⚠️
We pivoted to a logic-based **Reverse BFS**—starting from the single winning peg and working backward to find all winning configurations.
* **The Trap:** We ran into **State Space Inflation**. The algorithm found millions of states that were mathematically "solvable" (you could reach the end from them) but were **unreachable** from a standard full board.
* **Garden of Eden:** We were mapping the "Universal Solvable Space" rather than the actual game graph, wasting RAM on states that can never exist in a real game.

### Phase 3: The Hybrid HPC Solution (Success) ✅
To solve this exactly, we built a **Hybrid Forward-Pruned Reverse Solver**. The logic is simple set theory:
1.  **Forward Pass ($F$):** Map the reachable universe from the start. (~23.5M states).
2.  **Backward Pass ($B$):** Search backwards from the win, but *only* expand nodes that exist in $F$.
3.  **Intersection ($W = F \cap B$):** This leaves us with the true winning states.

---

##  Performance & Stats

By ditching pure Python for **Numba** and using the Hybrid approach, we achieved massive speedups.

| Metric | Pure Python BFS | **Machshevet (Numba Hybrid)** |
| :--- | :--- | :--- |
| **Execution Time** | Hours (or crash) | **~160 Seconds** |
| **Total Reachable States** | ~23,475,688 | **23,475,688** |
| **True Winning States** | Unknown | **1,679,072** (Canonical) |

### Under the Hood
* **Bitboards:** The 33-hole board is a single `uint64`. Move validation is $O(1)$ using bitwise masks.
* **Numba JIT:** Python is too slow for 23 million states. We compile the core logic to machine code, bypassing the GIL.
* **Symmetry Reduction:** The board has $D_4$ symmetry (8 rotations/reflections). We normalize every state to its "Canonical ID" (minimum integer value), effectively slashing memory usage by 8x.

---

##  The "Survival Funnel"

The GUI includes a real-time analytics graph that visualizes your mortality in the game.

* **The Funnel:** As you play, you see the number of possible winning paths dropping.
* **The Flatline:** If you make a move that leads to a dead end, the graph hits **Zero** instantly. You might still have valid moves left to play, but the Oracle knows you are already dead.



you can play it on https://hrsi56.pythonanywhere.com
