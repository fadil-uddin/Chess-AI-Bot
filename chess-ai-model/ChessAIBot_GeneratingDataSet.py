"""
Chess AI Dataset Generation Script


This script generates chess self-play data using the Stockfish engine and prepares
it for training the neural network. It performs the following:

1. Generates games where Stockfish plays against itself with some randomness.
2. Saves moves (in UCI format) and positions (in FEN) into `.npy` datasets.
3. Provides encoding functions for:
   - Chess boards into tensor representations.
   - Chess moves into integer indices (aligned with AlphaZero encoding).
4. Prepares encoded datasets for training.

Dependencies:
    - python-chess
    - stockfish
    - numpy
    - gym-chess (with AlphaZero extensions)

Author: [Syed Fadil Uddin]
Date: 2025
"""

# === Imports ===
import os
import glob
import random
import numpy as np
import chess
from stockfish import Stockfish
from gym_chess.alphazero.move_encoding import utils
import gym
from typing import List

# === Configuration ===
Num_Games = 500
STOCKFISH_PATH = "/Users/fadiluddin/Downloads/stockfish/stockfish-macos-m1-apple-silicon"
DATA_DIR = "/Users/fadiluddin/.spyder-py3/ChessData"
os.makedirs(DATA_DIR, exist_ok=True)

stockfish = Stockfish(path=STOCKFISH_PATH)


# ============================================================
#                   Helper Functions
# ============================================================
def check_end_condition(board: chess.Board) -> bool:
    """Return True if the game is over (checkmate, draw, etc.)."""
    return (
        board.is_checkmate()
        or board.is_stalemate()
        or board.is_insufficient_material()
        or board.can_claim_threefold_repetition()
        or board.can_claim_fifty_moves()
        or board.can_claim_draw()
    )


def find_next_idx() -> int:
    """Find the next available dataset index for saving new games."""
    files = glob.glob(os.path.join(DATA_DIR, "movesAndPositions*.npy"))
    if not files:
        return 1
    highest = max(
        int(f.split("movesAndPositions")[-1].split(".npy")[0]) for f in files
    )
    return highest + 1


def save_data(moves: List[str], positions: List[str]) -> None:
    """
    Save moves and positions as a numpy array.

    Args:
        moves: list of UCI move strings
        positions: list of FEN strings
    """
    moves = np.array(moves).reshape(-1, 1)
    positions = np.array(positions).reshape(-1, 1)
    moves_and_positions = np.concatenate((moves, positions), axis=1)

    next_idx = find_next_idx()
    out_path = os.path.join(DATA_DIR, f"movesAndPositions{next_idx}.npy")
    np.save(out_path, moves_and_positions)

    print("Saved successfully to:", out_path)


def latest_file() -> str:
    """Return the filename of the latest saved dataset."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, "movesAndPositions*.npy")))
    if not files:
        raise FileNotFoundError("No saved games found in ChessData.")
    return os.path.basename(files[-1])


def run_game(num_moves: int, filename: str = "movesAndPositions1.npy") -> chess.Board:
    """
    Replay a stored game up to `num_moves`.

    Args:
        num_moves: number of moves to replay
        filename: name of the .npy file with saved game

    Returns:
        A chess.Board object at the resulting state.
    """
    path = os.path.join(DATA_DIR, filename)
    game_data = np.load(path, allow_pickle=True)
    moves = game_data[:, 0]

    if num_moves > len(moves):
        print("Must enter <= game length. Game length here is:", len(moves))
        return None

    board = chess.Board()
    for mv in moves[:num_moves]:
        board.push(chess.Move.from_uci(str(mv)))
    return board


# ============================================================
#                   Game Generation
# ============================================================
def mine_games(num_games: int) -> None:
    """
    Generate `num_games` of Stockfish self-play with randomized top moves.
    Data is saved to disk in DATA_DIR.
    """
    MAX_MOVES = 500

    for _ in range(num_games):
        current_game_moves = []
        current_game_positions = []
        board = chess.Board()
        stockfish.set_position([])

        for _ in range(MAX_MOVES):
            top_moves = stockfish.get_top_moves(3)

            if not top_moves:
                print("Game ended.")
                break

            # Weighted random selection among Stockfish's best moves
            if len(top_moves) == 1:
                move = top_moves[0]["Move"]
            elif len(top_moves) == 2:
                move = random.choices(top_moves, weights=(80, 20), k=1)[0]["Move"]
            else:
                move = random.choices(top_moves, weights=(80, 15, 5), k=1)[0]["Move"]

            # Record position and move
            current_game_positions.append(stockfish.get_fen_position())
            board.push(chess.Move.from_uci(move))
            current_game_moves.append(move)
            stockfish.set_position(current_game_moves)

            if check_end_condition(board):
                break

        save_data(current_game_moves, current_game_positions)


# ============================================================
#                   Move Encoding
# ============================================================
def encode_knight(move: chess.Move) -> int:
    """Encode knight moves into AlphaZero-style action indices."""
    _TYPE_OFFSET = 56
    _DIRECTIONS = utils.IndexedTuple(
        (+2, +1), (+1, +2), (-1, +2), (-2, +1),
        (-2, -1), (-1, -2), (+1, -2), (+2, -1),
    )
    from_rank, from_file, to_rank, to_file = utils.unpack(move)
    delta = (to_rank - from_rank, to_file - from_file)
    if delta not in _DIRECTIONS:
        return None
    knight_move_type = _DIRECTIONS.index(delta)
    move_type = _TYPE_OFFSET + knight_move_type
    return np.ravel_multi_index((from_rank, from_file, move_type), dims=(8, 8, 73))


def encode_queen(move: chess.Move) -> int:
    """Encode queen (sliding) moves into AlphaZero-style action indices."""
    _DIRECTIONS = utils.IndexedTuple(
        (+1, 0), (+1, +1), (0, +1), (-1, +1),
        (-1, 0), (-1, -1), (0, -1), (+1, -1),
    )
    from_rank, from_file, to_rank, to_file = utils.unpack(move)
    delta = (to_rank - from_rank, to_file - from_file)

    is_queen_move = (
        (delta[0] == 0 or delta[1] == 0 or abs(delta[0]) == abs(delta[1]))
        and move.promotion in (chess.QUEEN, None)
    )
    if not is_queen_move:
        return None

    direction = tuple(np.sign(delta))
    distance = np.max(np.abs(delta))
    direction_idx = _DIRECTIONS.index(direction)
    distance_idx = distance - 1

    move_type = np.ravel_multi_index((direction_idx, distance_idx), dims=(8, 7))
    return np.ravel_multi_index((from_rank, from_file, move_type), dims=(8, 8, 73))


def encode_underpromotion(move: chess.Move) -> int:
    """Encode underpromotions (to knight, bishop, rook)."""
    _TYPE_OFFSET = 64
    _DIRECTIONS = utils.IndexedTuple(-1, 0, +1)
    _PROMOTIONS = utils.IndexedTuple(chess.KNIGHT, chess.BISHOP, chess.ROOK)
    from_rank, from_file, to_rank, to_file = utils.unpack(move)

    if not (move.promotion in _PROMOTIONS and from_rank == 6 and to_rank == 7):
        return None

    direction_idx = _DIRECTIONS.index(to_file - from_file)
    promotion_idx = _PROMOTIONS.index(move.promotion)

    under_type = np.ravel_multi_index((direction_idx, promotion_idx), dims=(3, 3))
    move_type = _TYPE_OFFSET + under_type
    return np.ravel_multi_index((from_rank, from_file, move_type), dims=(8, 8, 73))


def encode_move(move: str, board: chess.Board) -> int:
    """Encode a move string (UCI) given a board."""
    move = chess.Move.from_uci(move)
    if board.turn == chess.BLACK:
        move = utils.rotate(move)

    for encoder in (encode_queen, encode_knight, encode_underpromotion):
        action = encoder(move)
        if action is not None:
            return action
    raise ValueError(f"{move} is not a valid move")


def encode_board(board: chess.Board) -> np.ndarray:
    """Convert a chess board into a tensor (8x8x14)."""
    array = np.zeros((8, 8, 14), dtype=int)
    for square, piece in board.piece_map().items():
        rank, file = chess.square_rank(square), chess.square_file(square)
        idx = piece.piece_type - 1
        offset = 0 if piece.color == chess.WHITE else 6
        array[rank, file, idx + offset] = 1

    # Encode repetition counters
    array[:, :, 12] = board.is_repetition(2)
    array[:, :, 13] = board.is_repetition(3)
    return array


def encode_board_from_fen(fen: str) -> np.ndarray:
    """Convert a FEN string into encoded board tensor."""
    return encode_board(chess.Board(fen))


def encode_all_moves_and_positions():
    """Encode all saved games from ChessData into training-ready arrays."""
    files = os.listdir(DATA_DIR)
    os.makedirs("data/preparedData", exist_ok=True)

    for idx, fname in enumerate(files):
        raw = np.load(os.path.join(DATA_DIR, fname), allow_pickle=True)
        moves, fens = raw[:, 0], raw[:, 1]

        enc_moves, enc_pos = [], []
        for uci, fen in zip(moves, fens):
            board = chess.Board(fen)
            try:
                enc_moves.append(encode_move(uci, board))
                enc_pos.append(encode_board(board))
            except Exception as exc:
                print(f"[ENC ERR] {fname} ply {len(enc_moves)} |", exc)
                break

        np.save(f"data/preparedData/moves{idx}.npy", np.array(enc_moves))
        np.save(f"data/preparedData/positions{idx}.npy", np.array(enc_pos))


# ============================================================
#                   Main Execution
# ============================================================
if __name__ == "__main__":
    print(f"Generating {Num_Games} games...")
    mine_games(Num_Games)
    print("Encoding generated games...")
    encode_all_moves_and_positions()
    print("Done.")
