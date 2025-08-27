/**
 * App.js
 *
 * React frontend for the Chess AI Web Application.
 * Responsible for rendering the chessboard, tracking move history,
 * handling user interactions, and communicating with the Flask backend
 * which hosts the AI engine.
 *
 * Author: Syed Fadil Uddin
 */

import React, { useState } from "react";
import { Chessboard } from "react-chessboard";

function App() {
  // -------------------------
  // React State Hooks
  // -------------------------
  const [fen, setFen] = useState("start");          // Current board position in FEN notation
  const [history, setHistory] = useState([]);       // Move history array
  const [gameOver, setGameOver] = useState(false);  // Game over flag
  const [playerColor, setPlayerColor] = useState("w"); // Player's chosen color ("w" or "b")

  // -------------------------
  // Start a New Game
  // -------------------------
  async function newGame(color = "white") {
    try {
      const res = await fetch("/api/new_game", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ player: color }),
      });

      const data = await res.json();

      // Update state with new game data
      setFen(data.fen);
      setHistory(data.history || []);
      setGameOver(false);
      setPlayerColor(color === "white" ? "w" : "b");
    } catch (err) {
      console.error("Error starting new game:", err);
    }
  }

  // -------------------------
  // Handle Piece Movement
  // -------------------------
  async function onDrop(sourceSquare, targetSquare) {
    if (gameOver) return false;

    let moveUci = sourceSquare + targetSquare;

    // Handle pawn promotion automatically (default to queen)
    if (
      (sourceSquare[1] === "7" && targetSquare[1] === "8" && playerColor === "w") ||
      (sourceSquare[1] === "2" && targetSquare[1] === "1" && playerColor === "b")
    ) {
      moveUci += "q";
    }

    try {
      const res = await fetch("/api/move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ move: moveUci }),
      });

      const data = await res.json();

      if (res.ok && data.fen) {
        // Update board position and move history
        setFen(data.fen);
        setHistory(data.history || []);
        if (data.game_over) setGameOver(true);
        return true;
      } else {
        console.warn("Illegal move or error:", data);
        return false;
      }
    } catch (err) {
      console.error("Error making move:", err);
      return false;
    }
  }

  // -------------------------
  // Render Component
  // -------------------------
  return (
    <div
      style={{
        backgroundColor: "#2c3e50",
        height: "100vh",
        padding: "20px",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        color: "white",
      }}
    >
      <h1 style={{ marginBottom: "20px" }}>♟️ Chess AI App</h1>

      {/* --------------------- Controls Panel --------------------- */}
      <div
        style={{
          backgroundColor: "#1c1c1c",
          padding: "15px",
          borderRadius: "10px",
          marginBottom: "20px",
          boxShadow: "0px 4px 10px rgba(0,0,0,0.4)",
        }}
      >
        <button onClick={() => newGame("white")} style={{ marginRight: "10px" }}>
          New Game (You = White)
        </button>
        <button onClick={() => newGame("black")}>New Game (You = Black)</button>
      </div>

      <div style={{ display: "flex", gap: "20px" }}>
        {/* --------------------- Chessboard Panel --------------------- */}
        <div
          style={{
            backgroundColor: "#1c1c1c",
            padding: "15px",
            borderRadius: "10px",
            boxShadow: "0px 4px 10px rgba(0,0,0,0.4)",
          }}
        >
          <Chessboard
            position={fen}
            onPieceDrop={onDrop}
            boardWidth={500}
            boardOrientation={playerColor === "w" ? "white" : "black"}
          />
        </div>

        {/* --------------------- Move History Panel --------------------- */}
        <div
          style={{
            backgroundColor: "#1c1c1c",
            padding: "15px",
            borderRadius: "10px",
            width: "200px",
            boxShadow: "0px 4px 10px rgba(0,0,0,0.4)",
            overflowY: "auto",
            maxHeight: "500px",
          }}
        >
          <h3 style={{ marginBottom: "10px" }}>📜 Move History</h3>
          <ol>
            {history.map((entry, i) => (
              <li key={i}>
                <span style={{ color: entry.player === "You" ? "#00ffcc" : "#ffcc00" }}>
                  {entry.player}
                </span>
                : {entry.move}
              </li>
            ))}
          </ol>
          {gameOver && <h2 style={{ marginTop: "15px", color: "red" }}>Game Over</h2>}
        </div>
      </div>
    </div>
  );
}

export default App;
