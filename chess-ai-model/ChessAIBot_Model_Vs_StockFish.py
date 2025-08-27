"""
ChessAIBot_VS_StockFish.py


Play evaluation matches between the model and Stockfish.

Author: [Syed Fadil Uddin]
"""

import chess
import chess.engine
import torch
import numpy as np
from pathlib import Path
from gym_chess.alphazero.move_encoding import utils


# ================================================================
# Configuration
# ================================================================
MODEL_PATH = "/Users/fadiluddin/Projects/chess-ai-bot/chess-ai-bot-app/savedModels/last.pt"  # Path to trained model
STOCKFISH_PATH = "/Users/fadiluddin/Downloads/stockfish/stockfish-macos-m1-apple-silicon"
NUM_GAMES = 1                          # Number of games to play
STOCKFISH_ELO = 1320                   # Target Stockfish ELO
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


# ================================================================
# Model Definition
# ================================================================
class Residual(torch.nn.Module):
    """Residual block used inside the Policy-Value Network."""
    def __init__(self, c):
        super().__init__()
        self.c1 = torch.nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.b1 = torch.nn.BatchNorm2d(c)
        self.c2 = torch.nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.b2 = torch.nn.BatchNorm2d(c)

    def forward(self, x):
        y = torch.nn.functional.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return torch.nn.functional.relu(x + y)


class PolicyValueNet(torch.nn.Module):
    """
    AlphaZero-inspired convolutional neural network with:
    - Policy head (move probabilities)
    - Value head (position evaluation)
    """
    def __init__(self, planes=14, channels=96, nblocks=8, action_size=4672):
        super().__init__()
        # Initial convolution
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(planes, channels, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(channels),
            torch.nn.ReLU(inplace=True)
        )
        # Residual tower
        self.body = torch.nn.Sequential(*[Residual(channels) for _ in range(nblocks)])
        # Policy head
        self.p_head = torch.nn.Sequential(
            torch.nn.Conv2d(channels, 32, 1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True)
        )
        self.p_fc = torch.nn.Linear(32 * 8 * 8, action_size)
        # Value head
        self.v_head = torch.nn.Sequential(
            torch.nn.Conv2d(channels, 1, 1, bias=False),
            torch.nn.GroupNorm(1, 1),
            torch.nn.ReLU(inplace=True)
        )
        self.v_fc1 = torch.nn.Linear(8 * 8, 128)
        self.v_fc2 = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, H, W, C) -> (B, C, H, W)
        h = self.body(self.stem(x))

        # Policy output
        p = self.p_head(h).reshape(h.size(0), -1)
        p_out = self.p_fc(p)

        # Value output
        v = self.v_head(h).reshape(h.size(0), -1)
        v_out = torch.tanh(self.v_fc2(torch.nn.functional.relu(self.v_fc1(v), inplace=True))).squeeze(1)

        return p_out, v_out


# ================================================================
# Board Encoding
# ================================================================
def encode_board(board: chess.Board) -> np.ndarray:
    """
    Encode chess board into a (8, 8, 14) tensor.
    12 planes for pieces, 2 planes for repetition history.
    """
    arr = np.zeros((8, 8, 14), dtype=np.float32)
    for sq, pc in board.piece_map().items():
        r, f = chess.square_rank(sq), chess.square_file(sq)
        offset = 0 if pc.color == chess.WHITE else 6
        arr[r, f, pc.piece_type - 1 + offset] = 1.0
    arr[:, :, 12] = 1.0 if board.is_repetition(2) else 0.0
    arr[:, :, 13] = 1.0 if board.is_repetition(3) else 0.0
    return arr


# ================================================================
# Move Encoding
# ================================================================
def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """
    Encode a legal chess move into the model's action space index.
    Supports queen-like moves, knight jumps, and underpromotions.
    """
    if board.turn == chess.BLACK:
        move = utils.rotate(move)

    fr, ff = chess.square_rank(move.from_square), chess.square_file(move.from_square)
    tr, tf = chess.square_rank(move.to_square), chess.square_file(move.to_square)
    dr, df = np.sign(tr - fr), np.sign(tf - ff)

    # Queen-like moves
    if dr in (-1, 0, 1) and df in (-1, 0, 1) and not (dr == 0 and df == 0):
        dist = max(abs(tr - fr), abs(tf - ff))
        dir_idx = utils.IndexedTuple(
            (+1, 0), (+1, +1), (0, +1), (-1, +1),
            (-1, 0), (-1, -1), (0, -1), (+1, -1)
        ).index((dr, df))
        return fr * 8 * 73 + ff * 73 + dir_idx * 7 + (dist - 1)

    # Knight moves
    knight_dirs = utils.IndexedTuple(
        (+2, +1), (+1, +2), (-1, +2), (-2, +1),
        (-2, -1), (-1, -2), (+1, -2), (+2, -1)
    )
    if (dr, df) in knight_dirs:
        return fr * 8 * 73 + ff * 73 + 56 + knight_dirs.index((dr, df))

    # Underpromotions (non-queen)
    if move.promotion and move.promotion != chess.QUEEN:
        promo_dirs = utils.IndexedTuple(-1, 0, +1)
        promo_pieces = utils.IndexedTuple(chess.KNIGHT, chess.BISHOP, chess.ROOK)
        df_idx, promo_idx = promo_dirs.index(df), promo_pieces.index(move.promotion)
        return fr * 8 * 73 + ff * 73 + 64 + df_idx * 3 + promo_idx

    raise ValueError(f"Unencoded move: {move}")


# ================================================================
# Load Model
# ================================================================
model = PolicyValueNet().to(DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
model.eval()
print(f"✅ Loaded model from {MODEL_PATH}")


# ================================================================
# Stockfish Setup
# ================================================================
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
engine.configure({"UCI_LimitStrength": True, "UCI_Elo": STOCKFISH_ELO})


# ================================================================
# Play Matches
# ================================================================
results = []
for game_num in range(1, NUM_GAMES + 1):
    board = chess.Board()
    model_plays_white = (game_num % 2 == 1)  # Alternate colors

    while not board.is_game_over():
        if (board.turn == chess.WHITE and model_plays_white) or (board.turn == chess.BLACK and not model_plays_white):
            # Model move
            planes = encode_board(board)
            x = torch.tensor(planes, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits, _ = model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            move_scores = []
            for mv in board.legal_moves:
                try:
                    idx = move_to_index(mv, board)
                    move_scores.append((probs[idx], mv))
                except ValueError:
                    pass

            if not move_scores:
                break  # no encoded moves available

            best_move = max(move_scores, key=lambda t: t[0])[1]
            board.push(best_move)

        else:
            # Stockfish move
            result = engine.play(board, chess.engine.Limit(time=0.1))
            board.push(result.move)

    print(f"Game {game_num} ({'Model White' if model_plays_white else 'Model Black'}) → {board.result()}")
    results.append(board.result())


# ================================================================
# Summary
# ================================================================
wins = sum(
    (r == "1-0" and white) or (r == "0-1" and not white)
    for r, white in zip(results, [(i % 2 == 1) for i in range(1, NUM_GAMES + 1)])
)
losses = sum(
    (r == "0-1" and white) or (r == "1-0" and not white)
    for r, white in zip(results, [(i % 2 == 1) for i in range(1, NUM_GAMES + 1)])
)
draws = results.count("1/2-1/2")

print(f"\n📊 Model vs Stockfish @ {STOCKFISH_ELO} ELO")
print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")

engine.quit()
