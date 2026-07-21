# WontWork/ — Archived Phase 1 (AlphaZero / MCTS)

This directory is the leftover code from **Phase 1: The AI Overkill (Failure)**,
described in the [main README's "Story of Failure"](../README.md#phase-1-the-ai-overkill-failure).

Before landing on the Hybrid Forward-Pruned Reverse Solver that the rest of
this repo implements, the first approach was a from-scratch AlphaZero-style
pipeline: a neural network + Monte Carlo Tree Search, with reward shaping
based on Pagoda/topology functions instead of the sparse win/lose signal.
It never converged reliably on this board, so it was abandoned in favor of
the exact BFS-based solver.

## What's here

- `V2.py` … `V5.py`, `M2.py`, `mmm.py`, `tra.py`, `Machshevet.py` — successive
  iterations of the same AlphaZero attempt, kept in the order they were tried.
- `machshevetgame.ipynb` — a notebook used to experiment with the approach.
- `new/` — the most structured version of the pipeline: `Board.py`, `Game.py`,
  `NET.py` (the policy/value network), `MCTS.py`, `trainer.py`,
  `PegActionSpace.py`, `analyze_board_topology.py`,
  `reward_with_topology_analysis.py` (the Pagoda-style reward shaping).

## Status: unmaintained, kept for reference only

- Nothing in this folder is imported by, or required to run, the working
  solver/GUI/Flask code in the rest of the repo.
- It is **not** covered by [`tests/`](../tests) or the [CI workflow](../.github/workflows/ci.yml),
  and it is **not** included in [`requirements.txt`](../requirements.txt).
- It depends on packages the rest of the project doesn't need: `torch`,
  `torch_geometric`, `scipy`, `tqdm`, and optionally `mlflow`. None of these
  are installed by the project's setup instructions. If you want to run
  anything in here, install those yourself first.
- Code quality/consistency here is experimental-grade, not production-grade —
  treat it as a historical snapshot, not a starting point to build on without
  a deliberate decision to revive it.
