"""
ChessAIBot_EncodingData.py


This script encodes raw chess move/position data into a structured numerical
format suitable for training AI models. It converts:

1. **Moves (UCI format)** → into discrete integer encodings.
2. **Board states (FEN format)** → into multi-dimensional binary arrays.

Output is stored in `data/preparedData/` as `.npy` files, containing:
- Encoded moves: (N,)
- Encoded positions: (N, 8, 8, 14)

Author: [Syed Fadil Uddin]
"""

import os
import numpy as np
import gym
import chess
from gym_chess.alphazero.move_encoding import utils
from typing import List


# ============================================================
# Environment setup
# ============================================================
env = gym.make('ChessAlphaZero-v0')
env.reset()


# ============================================================
# Move Encoding Functions
# ============================================================
def encodeKnight(move: chess.Move) -> int | None:
    """
    Encode a knight move into a discrete action index.
    Returns None if not a knight move.
    """
    _TYPE_OFFSET: int = 56
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

    return np.ravel_multi_index(
        multi_index=(from_rank, from_file, move_type),
        dims=(8, 8, 73)
    )


def encodeQueen(move: chess.Move) -> int | None:
    """
    Encode a queen (or rook/bishop directional) move into a discrete action index.
    Returns None if not a valid queen move.
    """
    _DIRECTIONS = utils.IndexedTuple(
        (+1, 0), (+1, +1), (0, +1), (-1, +1),
        (-1, 0), (-1, -1), (0, -1), (+1, -1),
    )

    from_rank, from_file, to_rank, to_file = utils.unpack(move)
    delta = (to_rank - from_rank, to_file - from_file)

    is_horizontal = delta[0] == 0
    is_vertical = delta[1] == 0
    is_diagonal = abs(delta[0]) == abs(delta[1])
    is_promo = move.promotion in (chess.QUEEN, None)

    if not ((is_horizontal or is_vertical or is_diagonal) and is_promo):
        return None

    direction = tuple(np.sign(delta))
    distance = np.max(np.abs(delta))

    direction_idx = _DIRECTIONS.index(direction)
    distance_idx = distance - 1

    move_type = np.ravel_multi_index(
        multi_index=(direction_idx, distance_idx),
        dims=(8, 7)
    )

    return np.ravel_multi_index(
        multi_index=(from_rank, from_file, move_type),
        dims=(8, 8, 73)
    )


def encodeUnder(move: chess.Move) -> int | None:
    """
    Encode an underpromotion move (pawn → knight/bishop/rook).
    Returns None if not an underpromotion.
    """
    _TYPE_OFFSET: int = 64
    _DIRECTIONS = utils.IndexedTuple(-1, 0, +1)
    _PROMOTIONS = utils.IndexedTuple(chess.KNIGHT, chess.BISHOP, chess.ROOK)

    from_rank, from_file, to_rank, to_file = utils.unpack(move)

    if not (move.promotion in _PROMOTIONS and from_rank == 6 and to_rank == 7):
        return None

    direction_idx = _DIRECTIONS.index(to_file - from_file)
    promotion_idx = _PROMOTIONS.index(move.promotion)

    underpromotion_type = np.ravel_multi_index(
        multi_index=(direction_idx, promotion_idx),
        dims=(3, 3)
    )

    move_type = _TYPE_OFFSET + underpromotion_type

    return np.ravel_multi_index(
        multi_index=(from_rank, from_file, move_type),
        dims=(8, 8, 73)
    )


def encodeMove(move: str, board: chess.Board) -> int:
    """
    Encode a UCI chess move string into a discrete action index.

    Args:
        move: Move in UCI string format (e.g., 'e2e4').
        board: Current board object, required for rotation on black's turn.

    Returns:
        Integer action index.
    """
    move = chess.Move.from_uci(move)
    if board.turn == chess.BLACK:
        move = utils.rotate(move)

    for encoder in (encodeQueen, encodeKnight, encodeUnder):
        action = encoder(move)
        if action is not None:
            return action

    raise ValueError(f"Invalid move encoding: {move}")


# ============================================================
# Board Encoding Functions
# ============================================================
def encodeBoard(board: chess.Board) -> np.ndarray:
    """
    Encode a chess board into an (8, 8, 14) numpy array.

    Planes:
        - [0–5]   White pieces (P, N, B, R, Q, K)
        - [6–11]  Black pieces
        - [12]    Twofold repetition indicator
        - [13]    Threefold repetition indicator
    """
    array = np.zeros((8, 8, 14), dtype=int)

    for square, piece in board.piece_map().items():
        rank, file = chess.square_rank(square), chess.square_file(square)
        offset = 0 if piece.color == chess.WHITE else 6
        array[rank, file, piece.piece_type - 1 + offset] = 1

    array[:, :, 12] = board.is_repetition(2)
    array[:, :, 13] = board.is_repetition(3)

    return array


def encodeBoardFromFen(fen: str) -> np.ndarray:
    """
    Encode a board directly from its FEN string.
    """
    return encodeBoard(chess.Board(fen))


# ============================================================
# Bulk Encoding Pipeline
# ============================================================
def encodeAllMovesAndPositions(data_dir: str = "ChessData",
                               output_dir: str = "data/preparedData") -> None:
    """
    Encode all raw move/position files from ChessData into numpy arrays.

    Args:
        data_dir: Directory containing raw .npy move/position files.
        output_dir: Directory where encoded files will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    board = chess.Board()
    board.turn = chess.BLACK  # flip turn before first move

    files = os.listdir(data_dir)
    for idx, fname in enumerate(files):
        raw = np.load(os.path.join(data_dir, fname), allow_pickle=True)
        moves, fens = raw[:, 0], raw[:, 1]

        enc_moves, enc_positions = [], []

        for move, fen in zip(moves, fens):
            board.turn = not board.turn  # alternate turns
            try:
                enc_moves.append(encodeMove(move, board))
                enc_positions.append(encodeBoardFromFen(fen))
            except Exception:
                # Try flipping turn again if mismatch
                board.turn = not board.turn
                try:
                    enc_moves.append(encodeMove(move, board))
                    enc_positions.append(encodeBoardFromFen(fen))
                except Exception as e:
                    print(f"[ENC ERR] {fname} | move: {move} | FEN: {fen}")
                    print("Turn:", board.turn, "| idx:", len(enc_moves))
                    break

        np.save(os.path.join(output_dir, f"moves{idx}.npy"), np.array(enc_moves))
        np.save(os.path.join(output_dir, f"positions{idx}.npy"), np.array(enc_positions))

        print(f"[OK] Encoded {len(enc_moves)} moves from {fname}")


# ============================================================
# Main Execution
# ============================================================
if __name__ == "__main__":
    encodeAllMovesAndPositions()
