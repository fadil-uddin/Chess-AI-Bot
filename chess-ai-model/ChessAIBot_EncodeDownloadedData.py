"""
ChessAIBot_EncodeDownloadedData.py


This script converts downloaded PGN chess game data into encoded datasets for training 
the neural network-based chess AI. It extracts positions, legal moves, and evaluation labels 
using Stockfish, and stores them in NumPy arrays.

Features:
- Supports raw, .gz, .bz2, and .zst PGN formats.
- Encodes moves into AlphaZero-style action indices.
- Encodes board states as 8×8×14 feature planes.
- Uses Stockfish to assign evaluation values to positions.
- Writes encoded chunks of positions, moves, and values to disk.

Author: [Syed Fadil Uddin]
"""

# ================================================================
# Imports
# ================================================================
import os
import io
import re
import glob
import gzip
import bz2
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import chess
import chess.pgn
from stockfish import Stockfish

# Optional .zst support (pip install zstandard)
try:
    import zstandard as zstd
except ImportError:
    zstd = None

from gym_chess.alphazero.move_encoding import utils

# ================================================================
# Configuration
# ================================================================
PGN_DIR = "/Users/fadiluddin/RawDataSet"       # Folder with downloaded PGNs
OUT_DIR = Path("data/preparedData")            # Where encoded data is stored
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Opening balancing (disabled by default)
OPENING_PLIES = 6
CAP_PER_OPENING = None                         # None => keep all openings

# Position selection / limits
MAX_PLIES_PER_GAME = 500                       # Max plies per game
EVAL_EVERY_N_PLIES = 5                         # Assign Stockfish eval every Nth ply

# Stockfish settings
STOCKFISH_PATH = "/Users/fadiluddin/Downloads/stockfish/stockfish-macos-m1-apple-silicon"
SF_THREADS = 2
SF_HASH_MB = 64
SF_SKILL = 20
SF_DEPTH = 2                                   # Low for speed, higher = stronger labels

# File chunking
CHUNK_SIZE = 8_000                             # Positions per output file


# ================================================================
# Move Encoders
# ================================================================
def _enc_knight(move: chess.Move) -> int:
    """Encodes knight moves into AlphaZero action space."""
    TYPE_OFFSET = 56
    DIRS = utils.IndexedTuple(
        (+2,+1), (+1,+2), (-1,+2), (-2,+1),
        (-2,-1), (-1,-2), (+1,-2), (+2,-1)
    )
    fr, ff, tr, tf = utils.unpack(move)
    delta = (tr-fr, tf-ff)
    if delta not in DIRS:
        return None
    move_type = TYPE_OFFSET + DIRS.index(delta)
    return fr*8*73 + ff*73 + move_type


def _enc_queen(move: chess.Move) -> int:
    """Encodes queen (rook + bishop directions) moves into AlphaZero action space."""
    DIRS = utils.IndexedTuple(
        (+1,0), (+1,+1), (0,+1), (-1,+1),
        (-1,0), (-1,-1), (0,-1), (+1,-1)
    )
    fr, ff, tr, tf = utils.unpack(move)
    dr, df = tr-fr, tf-ff
    if not (dr == 0 or df == 0 or abs(dr) == abs(df)):
        return None
    if move.promotion not in (chess.QUEEN, None):
        return None

    direction = (np.sign(dr), np.sign(df))
    if direction not in DIRS:
        return None
    dist = max(abs(dr), abs(df))
    if not 1 <= dist <= 7:
        return None

    dir_idx = DIRS.index(direction)
    move_type = dir_idx*7 + (dist-1)
    return fr*8*73 + ff*73 + move_type


def _enc_under(move: chess.Move) -> int:
    """Encodes underpromotions (Knight, Bishop, Rook)."""
    TYPE_OFFSET = 64
    DIRS  = utils.IndexedTuple(-1, 0, +1)
    PROMO = utils.IndexedTuple(chess.KNIGHT, chess.BISHOP, chess.ROOK)

    fr, ff, tr, tf = utils.unpack(move)
    if fr != 6 or tr != 7 or move.promotion not in PROMO:
        return None

    df = tf - ff
    if df not in DIRS:
        return None

    under_type = DIRS.index(df)*3 + PROMO.index(move.promotion)
    move_type  = TYPE_OFFSET + under_type
    return fr*8*73 + ff*73 + move_type


# ============================================================
# Helper Functions
# ============================================================
def discover_pgns(pgn_dir: str) -> List[str]:
    """Find all PGN files in directory (supports compressed formats)."""
    paths = []
    for pat in ("*.pgn", "*.PGN", "*.pgn.zst", "*.pgn.gz", "*.pgn.bz2"):
        paths.extend(glob.glob(os.path.join(pgn_dir, pat)))
    return sorted(paths)


def open_pgn(path: str) -> io.TextIOBase:
    """Opens a PGN file, handling compression if needed."""
    if path.endswith(".zst"):
        if zstd is None:
            raise RuntimeError("Install zstandard to read .zst files (pip install zstandard).")
        fh = open(path, "rb")
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        stream = dctx.stream_reader(fh)
        return io.TextIOWrapper(stream, encoding="utf-8", errors="ignore")
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="ignore")
    if path.endswith(".bz2"):
        return io.TextIOWrapper(bz2.open(path, "rb"), encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


def encode_board(board: chess.Board) -> np.ndarray:
    """Encodes a board into 8×8×14 feature planes."""
    arr = np.zeros((8, 8, 14), dtype=np.int8)
    for sq, pc in board.piece_map().items():
        r, f = chess.square_rank(sq), chess.square_file(sq)
        off = 0 if pc.color == chess.WHITE else 6
        arr[r, f, pc.piece_type - 1 + off] = 1
    arr[:, :, 12] = board.is_repetition(2)
    arr[:, :, 13] = board.is_repetition(3)
    return arr


def encode_move_for_side(move: chess.Move, board: chess.Board) -> int:
    """
    Encodes move from White’s perspective (rotates board if Black to move).
    Tries queen, knight, and underpromotion encoders.
    """
    m = move if board.turn == chess.WHITE else utils.rotate(move)
    for fn in (_enc_queen, _enc_knight, _enc_under):
        idx = fn(m)
        if idx is not None:
            return idx
    raise ValueError(f"Cannot encode move {move.uci()}")


def opening_signature(game: chess.pgn.Game, k: int) -> str:
    """Return a unique signature for the first k plies of a game."""
    uci = []
    board = game.board()
    for i, mv in enumerate(game.mainline_moves()):
        if i >= k:
            break
        uci.append(mv.uci())
        board.push(mv)
    return " ".join(uci)


def normalize_eval_to_value(sf_eval: Dict) -> float:
    """Normalize Stockfish eval to [-1, 1] from POV of side-to-move."""
    typ = sf_eval.get("type")
    val = float(sf_eval.get("value", 0))
    if typ == "mate":
        cp = 20000.0 if val > 0 else -20000.0
    else:
        cp = val
    return float(np.tanh(cp / 600.0))


def next_pair_idx(out_dir: Path) -> int:
    """Finds next available dataset index for saving chunks."""
    files = glob.glob(str(out_dir / "moves*.npy"))
    idxs = [int(m.group(1)) for f in files if (m := re.search(r"moves(\d+)\.npy$", f))]
    return (max(idxs) + 1) if idxs else 0


def write_chunk(idx: int, X: List[np.ndarray], Y: List[int], V: List[float]):
    """Writes a chunk of data to disk."""
    np.save(OUT_DIR / f"positions{idx}.npy", np.asarray(X, dtype=np.int8))
    np.save(OUT_DIR / f"moves{idx}.npy", np.asarray(Y, dtype=np.int32))
    np.save(OUT_DIR / f"values{idx}.npy", np.asarray(V, dtype=np.float32))
    print(f"✓ Saved chunk {idx}: {len(Y)} positions")


# ================================================================
# Main Conversion
# ================================================================
def convert():
    """Main loop: process PGNs, encode positions & moves, save chunks."""
    pgn_paths = discover_pgns(PGN_DIR)
    print(f"Found {len(pgn_paths)} PGN files")
    if not pgn_paths:
        print("⚠️ No PGN files found in PGN_DIR.")
        return

    sf = Stockfish(
        path=STOCKFISH_PATH,
        depth=SF_DEPTH,
        parameters={"Threads": SF_THREADS, "Hash": SF_HASH_MB, "Skill Level": SF_SKILL},
    )

    pair_idx = next_pair_idx(OUT_DIR)
    X_buf, Y_buf, V_buf = [], [], []
    open_count: Dict[str, int] = {}

    total_games = kept_games = total_positions = 0

    for path in pgn_paths:
        with open_pgn(path) as fh:
            while (game := chess.pgn.read_game(fh)) is not None:
                total_games += 1
                sig = opening_signature(game, OPENING_PLIES)
                if CAP_PER_OPENING and open_count.get(sig, 0) >= CAP_PER_OPENING:
                    continue

                board = game.board()
                plies, any_kept = 0, False

                for mv in game.mainline_moves():
                    if plies >= MAX_PLIES_PER_GAME:
                        break

                    # Try to encode BEFORE applying the move
                    try:
                        y = encode_move_for_side(mv, board)
                    except Exception:
                        board.push(mv)
                        plies += 1
                        continue

                    X = encode_board(board)

                    v = 0.0
                    if (plies % EVAL_EVERY_N_PLIES) == 0:
                        try:
                            sf.set_fen_position(board.fen())
                            v = normalize_eval_to_value(sf.get_evaluation())
                        except Exception:
                            v = 0.0

                    X_buf.append(X); Y_buf.append(y); V_buf.append(v)
                    total_positions += 1
                    open_count[sig] = open_count.get(sig, 0) + 1
                    any_kept = True

                    plies += 1
                    board.push(mv)

                    if len(X_buf) >= CHUNK_SIZE:
                        write_chunk(pair_idx, X_buf, Y_buf, V_buf)
                        pair_idx += 1
                        X_buf.clear(); Y_buf.clear(); V_buf.clear()

                if any_kept:
                    kept_games += 1

    if X_buf:
        write_chunk(pair_idx, X_buf, Y_buf, V_buf)

    print(f"✅ Done. Games={total_games}, Kept={kept_games}, "
          f"Positions={total_positions}, Openings={len(open_count)}")


# ============================================================
# Main Execution
# ============================================================
if __name__ == "__main__":
    convert()
