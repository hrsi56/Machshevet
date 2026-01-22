# Machshevet (Peg Solitaire) Solver & Analytics 🧠

> **An optimized Hybrid-BFS Oracle comprising ~3 million unique canonical states.**

This project implements a high-performance solver and analysis tool for the classic **Peg Solitaire (Standard English Board)**. It features a real-time "Survival Funnel" dashboard that visualizes the exact number of winning paths available to the player at any given moment.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Algorithm](https://img.shields.io/badge/Algorithm-Hybrid_Forward_Backward_BFS-purple)
![State Space](https://img.shields.io/badge/States-~3M_Canonical-success)

## 🚀 The Algorithmic Journey

This repository documents a journey of optimization, moving from "Heavy AI" to precise mathematical graph theory.

### Phase 1: The AI Attempt (Failed)
Initially, we implemented **AlphaZero** (Deep Reinforcement Learning with MCTS).
* **Lesson Learned:** For a deterministic puzzle with a finite state space (~33 bits), Neural Networks are overkill. They approximate the solution, whereas we needed exactness. Also, the reward signal is too sparse for efficient training.

### Phase 2: Naive Reverse BFS (Inefficient)
We switched to a **Reverse BFS** approach—starting from the single winning peg and working backwards.
* **The Problem:** The algorithm found **13+ million** solvable states.
* **Why:** It mapped the "Universal Solvable Space"—including board configurations that are mathematically impossible to reach from the standard starting game.

### Phase 3: The Hybrid Solution (Current) ✅
To solve *exactly* the standard game, we implemented a **Forward-Pruned Reverse Solver**:
1.  **Forward Pass:** Map all states reachable from the start (The "Reachable Universe").
2.  **Backward Pass:** Map winning states starting from the end, **but strictly intersecting** with the Reachable Universe.
* **Result:** The search space collapsed from 13M to exactly **~3 million** valid canonical states.

## 🛠️ Technical Architecture

The engine runs purely on the CPU using highly optimized techniques:

* **Hybrid Search Strategy:** Combines reachability analysis with reverse solving to eliminate unreachable states.
* **Bitboards:** The board is represented as a single 64-bit integer. Move validation and application are bitwise operations ($O(1)$).
* **Symmetry Pruning:** Every board state is normalized to its "Canonical Form" (minimum value of 8 rotations/reflections), reducing memory usage by factor of ~8.
* **O(1) Lookup:** During gameplay, the AI checks a hash map. If the current state exists in the map, it is a guaranteed win.

## 📉 The "Survival Funnel"

The UI features a unique analytics graph:
* **Start:** Shows thousands of winning options.
* **Mid-Game:** The graph creates a "funnel" shape as the decision space narrows.
* **Dead End:** If the player makes a fatal mistake, the graph flatlines to **Zero** instantly, providing immediate feedback.

## 📂 Repository Structure

### ✅ The Production Engine
* **`solver.py`**: The main executable. Contains the Hybrid Engine, Bitboard logic, and Tkinter GUI.
* **`solitaire_standard_brain.pkl`**: (Generated on first run) The optimized "brain" file containing the map of ~3M winning states.

### 🧪 The Research Archive (`/WontWork`)
A collection of previous attempts, kept for educational purposes and documentation of the development process.
* **`MCTS.py` / `NET.py` / `trainer.py`**: The deprecated AlphaZero implementation.
* **`Legacy Solvers`**: Earlier brute-force attempts.

## ⚡ Performance Benchmarks

| Metric | Naive Reverse BFS | **Hybrid Forward-Backward (Current)** |
| :--- | :--- | :--- |
| **Total States Mapped** | > 13,000,000 | **~2,900,000** |
| **Logic** | Universal Solvability | **Standard Game Solvability** |
| **Training Time** | ~5-10 Minutes | **~30 Seconds** |
| **Inference Time** | Instant | **Instant** |

## 🎮 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/machshevet-solver.git](https://github.com/your-username/machshevet-solver.git)
    cd machshevet-solver
    ```

2.  **Run the application:**
    ```bash
    python solver.py
    ```
    *Note: On the first run, the system will perform the "Hybrid Training" process to map the game graph. This takes about 30-40 seconds.*

3.  **Controls:**
    * **Left Click:** Move peg.
    * **Auto Solve:** Watch the AI execute a perfect solution from your current state.
    * **Undo:** Revert move (visualize how the "Funnel" recovers).

---
**Author:** Yarden Viktor Dejorno
