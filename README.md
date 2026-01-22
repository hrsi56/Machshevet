# Machshevet (Peg Solitaire) Solver & Analysis 

This repository documents the journey of solving the classic **Peg Solitaire (English Board)** game. It represents a pivot from complex Deep Reinforcement Learning to optimized classical algorithms.

The project currently features a **Perfect Play Oracle** capable of solving any board configuration instantly, alongside a "Survival Funnel" visualization that shows the player exactly how many winning paths remain in real-time.

## Simpler is Better

This project began with a heavy artillery approach: implementing **AlphaZero** (MCTS + ResNets) to "learn" the game. While educational, we realized that for a deterministic puzzle of this size (~33 holes), Reinforcement Learning is overkill and inefficient.

**Why RL failed (or was inefficient) here:**
* **Sparse Rewards:** The game has only one winning state out of billions.
* **Determinism:** There is no luck involved. Approximating the solution with probabilities (Neural Networks) is less accurate than calculating the exact graph.

**The Solution (`solver.py`):**
We switched to **Reverse Breadth-First Search (BFS)** combined with **Bitboards**.
Instead of searching for a needle in a haystack (start $\to$ win), we start from the needle (the winning state) and work backwards to find every possible valid board configuration.

##  Key Features (`solver.py`)

The current engine runs on the CPU and is optimized for extreme speed:

* **Reverse Engineering:** Maps the entire game tree from the winning state backwards. It only visits states that *can* win.
* **Bitwise Operations:** The board is represented as a single 64-bit integer, making move validation and application take nanoseconds.
* **Symmetry Pruning:** Normalizes every board state to its canonical form (considering 8 rotational and reflectional symmetries), reducing the search space by ~8x.
* **O(1) Lookup:** During gameplay, the AI doesn't "think." It checks a hash map of ~12 million pre-calculated states. If the state exists, it's a guaranteed win.
* **Survival Funnel Graph:** A real-time dashboard showing the number of winning paths dropping as you make moves, creating a visual representation of the game's difficulty.

##  Repository Structure

###  The Working Solution
* **`solver.py`**: The main executable. Contains the Bitboard engine, the Reverse BFS trainer, and the Tkinter GUI.
* **`solitaire_fast_brain.pkl`**: (Generated upon first run) The "brain" file containing the map of all winning states.

###  The Research Lab (`/WontWork`)
This folder contains the original AlphaZero implementation. While these algorithms are powerful, they were not the right tool for *this specific* job. They are kept for documentation and educational purposes.

* **`MCTS.py`**: A full Monte Carlo Tree Search implementation with PUCT and Dirichlet noise.
* **`NET.py`**: A ResNet-based Neural Network (PyTorch) with Policy and Value heads.
* **`trainer.py`**: The training loop for self-play and backpropagation.
* **`analyze_board_topology.py`**: An attempt to create heuristic rewards based on board fragmentation and "patterns of death."

## ⚡ Performance

* **Training Time:** ~30-60 seconds (on a standard CPU).
* **Memory:** Maps approx. 10-12 million unique canonical winning states.
* **Inference:** Instantaneous (Dictionary lookup).


**Yarden Viktor Dejorno**
