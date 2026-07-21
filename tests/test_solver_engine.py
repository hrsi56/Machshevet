"""Fast, deterministic tests for the canonical solver engine (Parallel_Solver_Numba).

These only exercise board geometry, move generation, symmetry handling, and
memory persistence -- none of them run a real solve, so the full suite stays
under a second. The end-to-end `train()` regression lives in
test_solver_engine_slow.py (marked `slow`, excluded by default).
"""
import pytest

import Parallel_Solver_Numba as engine


@pytest.fixture(scope="module")
def solver():
    """A single shared, untrained solver instance for read-only geometry checks."""
    return engine.PegSolitaireSolver()


def test_board_has_33_holes(solver):
    assert len(solver.r_c_to_bit) == 33
    assert bin(solver.valid_mask).count("1") == 33


def test_initial_board_leaves_only_center_empty(solver):
    initial = solver.get_initial_board()
    assert initial == solver.valid_mask & ~(1 << solver.center_bit)
    assert (initial >> solver.center_bit) & 1 == 0
    assert bin(initial).count("1") == 32


def test_center_hole_is_board_center(solver):
    assert solver.bit_to_r_c[solver.center_bit] == (3, 3)


def test_move_count_matches_standard_english_board(solver):
    # The 33-hole English cross board has exactly 76 legal jump moves.
    assert len(solver.moves_metadata) == 76
    assert len(solver.moves_mask) == 76


def test_every_move_touches_exactly_three_holes(solver):
    for mask in solver.moves_mask:
        assert bin(int(mask)).count("1") == 3


def test_reverse_moves_mirror_forward_moves(solver):
    assert len(solver.reverse_moves) == len(solver.moves_metadata)
    for fwd, rev in zip(solver.moves_metadata, solver.reverse_moves):
        assert rev["mask"] == fwd["mask"]
        assert rev["req_peg"] == fwd["check_dst"]
        assert rev["req_empty"] == fwd["check_src"]


def test_symmetry_maps_are_bijections_over_all_holes(solver):
    all_bits = set(solver.bit_to_r_c)
    assert len(solver.symmetry_maps) == 8
    for mapping in solver.symmetry_maps:
        assert set(mapping.keys()) == all_bits
        assert set(mapping.values()) == all_bits  # bijection: no collisions


def test_symmetry_maps_include_identity(solver):
    assert any(
        all(src == dst for src, dst in mapping.items())
        for mapping in solver.symmetry_maps
    )


def test_get_canonical_is_idempotent(solver):
    board = solver.get_initial_board()
    canonical = solver.get_canonical(board)
    assert solver.get_canonical(canonical) == canonical


def test_get_canonical_never_exceeds_input(solver):
    board = solver.get_initial_board()
    assert solver.get_canonical(board) <= board


def test_get_canonical_is_symmetry_invariant(solver):
    board = solver.get_initial_board()
    canonical = solver.get_canonical(board)

    for mapping in solver.symmetry_maps:
        rotated = 0
        remaining = board
        while remaining:
            lsb = remaining & -remaining
            idx = lsb.bit_length() - 1
            rotated |= 1 << mapping[idx]
            remaining ^= lsb
        assert solver.get_canonical(rotated) == canonical


def _fresh_solver(monkeypatch, tmp_path, filename="brain.pkl"):
    # MEMORY_FILE is read during __init__, so it must be patched on the class
    # before construction -- and pointed at an isolated tmp file so tests never
    # touch (or depend on) a real brain file on disk.
    monkeypatch.setattr(engine.PegSolitaireSolver, "MEMORY_FILE", str(tmp_path / filename))
    return engine.PegSolitaireSolver()


def test_fresh_instance_has_no_precomputed_solution(monkeypatch, tmp_path):
    fresh = _fresh_solver(monkeypatch, tmp_path)
    assert fresh.loaded_from_disk is False
    assert fresh.winning_states == set()


def test_save_and_load_memory_roundtrip(monkeypatch, tmp_path):
    fresh = _fresh_solver(monkeypatch, tmp_path)
    fresh.winning_states = {1, 2, 3, 5, 8}
    fresh.save_memory()

    reloaded = _fresh_solver(monkeypatch, tmp_path)
    assert reloaded.loaded_from_disk is True
    assert reloaded.winning_states == {1, 2, 3, 5, 8}
