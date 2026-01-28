import os
import secrets
from flask import Flask, render_template, jsonify, request, session
from solver_logic import PegSolitaireSolver

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

print(f"Loading solver in: {os.getcwd()}")
solver = PegSolitaireSolver()

if solver.brain_loaded:
    print("✅ BRAIN LOADED SUCCESSFULLY!")
else:
    print(f"❌ WARNING: Brain file not found at: {solver.brain_path}")


def board_to_grid(board_val):
    grid = []
    for r in range(7):
        row_data = []
        for c in range(7):
            if (r, c) in solver.r_c_to_bit:
                idx = solver.r_c_to_bit[(r, c)]
                has_peg = (board_val >> idx) & 1
                row_data.append(1 if has_peg else 0)
            else:
                row_data.append(-1)
        grid.append(row_data)
    return grid


@app.route('/')
def index():
    if 'board' not in session:
        session['board'] = solver.get_initial_board()
        session['history'] = []
    return render_template('index.html')


@app.route('/api/state', methods=['GET'])
def get_state():
    board = session.get('board', solver.get_initial_board())
    winning_count = solver.get_winning_moves_count(board)

    return jsonify({
        'grid': board_to_grid(board),
        'winning_moves': winning_count,
        'brain_active': solver.brain_loaded,
        'game_over': winning_count == 0 and board != (1 << solver.center_bit),
        'victory': board == (1 << solver.center_bit)
    })


@app.route('/api/heatmap', methods=['POST'])
def get_heatmap():
    board = session.get('board')
    data = request.json

    try:
        sr = int(data.get('r'))
        sc = int(data.get('c'))
    except (ValueError, TypeError):
        return jsonify([])

    if (sr, sc) not in solver.r_c_to_bit:
        return jsonify([])

    src_idx = solver.r_c_to_bit[(sr, sc)]
    heatmap = []

    for m in solver.moves:
        if m['src'] == src_idx:
            if (board & m['check_src'] == m['check_src']) and (board & m['check_dst'] == 0):
                next_board = board ^ m['mask']

                future_moves = 0
                is_safe = False

                if next_board == (1 << solver.center_bit):
                    is_safe = True
                    future_moves = 99
                elif solver.brain_loaded:
                    if solver.get_canonical(next_board) in solver.winning_states:
                        is_safe = True
                        future_moves = solver.get_winning_moves_count(next_board)
                else:
                    future_moves = -1

                dst_r, dst_c = solver.bit_to_r_c[m['dst']]
                heatmap.append({
                    'r': dst_r, 'c': dst_c,
                    'count': future_moves,
                    'safe': is_safe
                })

    return jsonify(heatmap)


@app.route('/api/move', methods=['POST'])
def make_move():
    board = session.get('board')
    history = session.get('history', [])
    data = request.json

    move_key = data.get('move_key')
    target_move = None

    if move_key:
        for m in solver.moves:
            if m['move_key'] == move_key:
                target_move = m
                break
    else:
        sr, sc = int(data['src'][0]), int(data['src'][1])
        dr, dc = int(data['dst'][0]), int(data['dst'][1])

        if (sr, sc) in solver.r_c_to_bit and (dr, dc) in solver.r_c_to_bit:
            s_idx = solver.r_c_to_bit[(sr, sc)]
            d_idx = solver.r_c_to_bit[(dr, dc)]
            for m in solver.moves:
                if m['src'] == s_idx and m['dst'] == d_idx:
                    target_move = m
                    break

    if target_move:
        if (board & target_move['check_src'] == target_move['check_src']) and \
           (board & target_move['check_dst'] == 0):
            history.append(board)
            session['history'] = history
            board ^= target_move['mask']
            session['board'] = board
            return jsonify({'success': True})

    return jsonify({'success': False})


@app.route('/api/undo', methods=['POST'])
def undo():
    history = session.get('history', [])
    if history:
        prev = history.pop()
        session['board'] = prev
        session['history'] = history
        return jsonify({'success': True})
    return jsonify({'success': False})


@app.route('/api/reset', methods=['POST'])
def reset():
    session['board'] = solver.get_initial_board()
    session['history'] = []
    return jsonify({'success': True})


@app.route('/api/solve', methods=['POST'])
def solve():
    board = session.get('board')
    path = solver.solve_full_path(board)
    if path:
        return jsonify({'solved': True, 'path': path})
    return jsonify({'solved': False})