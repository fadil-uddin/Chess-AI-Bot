"""
Flask Backend for Chess AI Web App

This script serves as the backend for the web-based chess application where 
you can play against the custom-trained chess AI.

Author: [Syed Fadil Uddin]
"""


import os
import torch
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import chess
from model import PolicyValueNet  # <-- your model class

# ---------------- Flask setup ----------------
app = Flask(__name__, static_folder="static", static_url_path="/")
CORS(app)

# ---------------- Load AI model ----------------
MODEL_PATH = os.path.join("savedModels", "model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PolicyValueNet()
if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if "model" in checkpoint:  # full training checkpoint
        model.load_state_dict(checkpoint["model"])
    else:  # raw state_dict
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print(f"✅ Loaded model from {MODEL_PATH}")
else:
    print(f"⚠️ No model found at {MODEL_PATH}, AI will make random moves")

# ---------------- Game state ----------------
board = chess.Board()
move_history = []  # list of {move: "e2e4", player: "White"}

def board_to_tensor(board):
    """
    Convert python-chess board into (8,8,14) tensor for the model.
    """
    planes = np.zeros((8, 8, 14), dtype=np.float32)
    piece_map = board.piece_map()

    for square, piece in piece_map.items():
        r = 7 - (square // 8)
        c = square % 8
        idx = piece.piece_type - 1
        if piece.color == chess.BLACK:
            idx += 6
        planes[r, c, idx] = 1.0

    if board.turn == chess.WHITE:
        planes[:, :, 12] = 1.0
    else:
        planes[:, :, 13] = 1.0

    return planes

def get_ai_move():
    """
    Generate AI move using the trained model.
    """
    if board.is_game_over():
        return None

    tensor = torch.tensor(board_to_tensor(board)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    move_scores = []
    for move in legal_moves:
        uci = move.uci()
        move_idx = abs(hash(uci)) % probs.shape[0]
        move_scores.append((probs[move_idx], move))

    move_scores.sort(reverse=True, key=lambda x: x[0])
    return move_scores[0][1]

# ---------------- API endpoints ----------------
@app.route("/api/new_game", methods=["POST"])
def new_game():
    global board, move_history
    data = request.get_json() or {}
    player_color = data.get("player", "white")

    board = chess.Board()
    move_history = []

    ai_move = None
    # If user chooses black, AI moves first
    if player_color == "black":
        move = get_ai_move()
        if move:
            board.push(move)
            move_history.append({"player": "AI", "move": move.uci()})
            ai_move = move.uci()

    return jsonify({
        "status": "ok",
        "fen": board.fen(),
        "history": move_history,
        "ai_move": ai_move
    })

@app.route("/api/move", methods=["POST"])
def player_move():
    global board, move_history
    data = request.json
    move_uci = data.get("move")

    try:
        move = chess.Move.from_uci(move_uci)
        if move in board.legal_moves:
            board.push(move)
            move_history.append({"player": "Human", "move": move.uci()})

            ai_move = None
            if not board.is_game_over():
                ai = get_ai_move()
                if ai:
                    board.push(ai)
                    move_history.append({"player": "AI", "move": ai.uci()})
                    ai_move = ai.uci()

            return jsonify({
                "status": "ok",
                "fen": board.fen(),
                "history": move_history,
                "ai_move": ai_move,
                "game_over": board.is_game_over()
            })
        else:
            return jsonify({"status": "error", "message": "Illegal move"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/api/state", methods=["GET"])
def game_state():
    return jsonify({
        "fen": board.fen(),
        "turn": "white" if board.turn else "black",
        "is_game_over": board.is_game_over(),
        "history": move_history
    })

# ---------------- Serve React frontend ----------------
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(app.static_folder, path)

# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
