"""End-to-end regression test for the solver's actual search algorithm.

Runs a full solver.train() over the real ~23M-state reachable space and
checks the result against the numbers documented in README.md's
"Performance Benchmarks" table. Where test_solver_engine.py checks the
scaffolding (geometry, symmetry, persistence), this checks the algorithm
itself actually finds the right answer.

Excluded from the default `pytest` run (see the `addopts` in pyproject.toml).
Run explicitly with `pytest -m slow`. Takes on the order of a minute or more
depending on CPU core count -- see the CI workflow's separate "slow" job.
"""
import pytest

import Parallel_Solver_Numba as engine

# From README.md's "Performance Benchmarks" table.
EXPECTED_WINNING_STATES = 1_679_072


@pytest.fixture(scope="module")
def trained_solver(tmp_path_factory):
    original_memory_file = engine.PegSolitaireSolver.MEMORY_FILE
    brain_path = tmp_path_factory.mktemp("brain") / "brain.pkl"
    engine.PegSolitaireSolver.MEMORY_FILE = str(brain_path)
    try:
        solver = engine.PegSolitaireSolver()
        assert solver.loaded_from_disk is False  # sanity: this is a real run, not a cached load
        solver.train()
        yield solver, brain_path
    finally:
        engine.PegSolitaireSolver.MEMORY_FILE = original_memory_file


@pytest.mark.slow
def test_train_reproduces_documented_winning_state_count(trained_solver):
    solver, brain_path = trained_solver
    assert len(solver.winning_states) == EXPECTED_WINNING_STATES
    assert brain_path.exists()  # train() should persist the brain via save_memory()


@pytest.mark.slow
def test_solve_full_path_reaches_the_single_peg_win(trained_solver):
    solver, _ = trained_solver
    start = solver.get_initial_board()
    path = solver.solve_full_path(start)

    assert path is not None
    assert path[0] == start
    assert path[-1] == (1 << solver.center_bit)

    legal_diffs = {int(m) for m in solver.moves_mask}
    for before, after in zip(path, path[1:]):
        assert (before ^ after) in legal_diffs  # every step is one legal jump
