# ♟️ Chess AI Bot

A custom chess AI powered by a convolutional neural network, trained on **~100M chess positions** from Stockfish self-play and public datasets from Lichess.  
The model reaches an estimated **~1100 Elo** (16% win-rate vs Stockfish-3 at Elo 1320) and is playable via a full-stack **React + Flask web app** in real time, complete with a virtual board visualisation, the ability to start a game as black or white, and game move history.

---

## 📑 Table of Contents

- [Demo](#-demo)  
- [Motivation](#-motivation)  
- [Features](#-features)  
- [Tech Stack](#-tech-stack)  
- [Quickstart (Run Locally)](#-quickstart-run-locally)  
  - [Recommended: Docker](#recommended-docker)  
  - [Without Docker (venv)](#without-docker-venv)  
- [Model Export (Reduce Checkpoint Size)](#-model-export-reduce-checkpoint-size)  
- [API Reference](#-api-reference)  
- [Project Layout](#-project-layout)  
- [Model & Training Summary](#-model--training-summary)  
- [Reproducibility & Scripts](#-reproducibility--scripts)  
- [Roadmap / Future Work](#-roadmap-of-future-work)  
- [Acknowledgements](#-acknowledgements)  
- [License](#-license)  
- [Author / Contact](#-author-)

---

## 📸 Demo
![White vs AI](https://github.com/user-attachments/assets/023c8ff5-afca-46e7-9321-fadce1b35860) 

[Shows gameplay as White against the Chess AI Bot]


![Black vs AI](https://github.com/user-attachments/assets/a4f34b4e-07b5-46a5-bfbe-f4b4493f48e5)

[Shows gameplay as Black against the Chess AI Bot]

---

## 🎯 Motivation
I built this project to deepen my understanding of **machine learning, reinforcement learning, and game theory**.  
Since I’m interested in strategy games, creating a chess AI from scratch allowed me to combine theory with a practical, challenging application and deploy it as a playable web app.

---

## ⚡ Features

**AI / Training**
- Residual CNN policy + value network suitable for chess move prediction and outcome estimation.
- Trains from chunked NumPy datasets (`positions*.npy`, `moves*.npy`, `values*.npy`) produced by encoder scripts.
- Uses Stockfish evaluations as value labels for supervised training.
- Memory-mapped dataset loader with an LRU cache for efficient training on large corpora.
- Checkpointing, resume training, TensorBoard logging.

**Web App**
- React frontend with an interactive chessboard and move validation.
- Flask backend hosting the model for inference (PyTorch).
- Full move history, New Game options (choose color), and clean UI.
- Dockerised for reproducible deployment.

---

## 🛠 Tech Stack
**AI / ML**: PyTorch, python-chess, Stockfish, NumPy, Python  
**Backend**: Flask (Python)  
**Frontend**: React, Chessboard.js  
**Deployment**: Docker  

---

## 🚀 Quickstart (Run Locally)

Clone the repository:

```bash
git clone https://github.com/yourusername/chess-ai-bot.git
cd chess-ai-bot
```

### Recommended: Docker

```bash
docker build -t chess-ai-bot .
docker run -p 5001:5000 chess-ai-bot
```

➡️ The app will be available at:  
👉 http://localhost:5001

### Without Docker (venv)

**Backend**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

**Frontend**
```bash
cd frontend
npm install
npm start
```

---

## 📦 Model Export (Reduce Checkpoint Size)

Training produces large checkpoints (`last.pt`, often hundreds of MB).  
For deployment, convert them to a smaller `model.pth`:

```bash
python ChessAIBot_ExportModel.py --checkpoint savedModels/last.pt --output model.pth
```

---

## 🔌 API Reference

The Flask backend exposes simple endpoints:

- `POST /predict`  
  Input: JSON with FEN (chess position).  
  Output: Predicted best move(s) + value evaluation.  

- `GET /health`  
  Returns `200 OK` if the model is loaded.

---

## 📂 Project Layout

```
chess-ai-bot/
│──chess-ai-app
│    │── backend/             # Flask backend serving the AI
│    │   ├── app.py           # Flask app entrypoint
│    │   ├── model_loader.py  # Loads PyTorch model
│    │   └── requirements.txt # Python dependencies
│    │
│    │── frontend/            # React frontend
│    │   ├── src/
│    │   └── package.json
│    │
│    │── savedModels/         # Training checkpoints (e.g. last.pt, model.pth)
│    │── Dockerfile
│ 
│── chess-ai-model
│    │── ChessAIBot_EncodeDownloadedData.py   # Encodes raw PGN datasets into NumPy arrays
│    │── ChessAIBot_EncodingData.py           # Converts chess positions into tensors (input planes)
│    │── ChessAIBot_ExportModel.py            # Exports large training checkpoints into smaller .pth files
│    │── ChessAIBot_GeneratingDataset.py      # Generates dataset from PGN files using python-chess + Stockfish
│    │── ChessAIBot_Model_Vs_Stockfish.py     # Evaluates trained model by playing matches vs Stockfish
│    │── ChessAIBot_Model.py                  # Defines the neural network architecture (policy + value CNN)
│    │── ChessAIBot_TrainingArchitecture.py   # Training loop, optimizer, logging, checkpoint saving
│
│── README.md
```

---

## 🧠 Model & Training Summary

- **Architecture:** Convolutional Residual Network (8 residual blocks, policy + value heads).  
- **Dataset:** ~100M chess positions (Stockfish self-play + public games).  
- **Training:** Supervised learning on move labels + outcome prediction.  
- **RL Concepts:** Incorporated reinforcement learning concepts for self-play decision making.  
- **Export:** Checkpoints (`last.pt`) converted into smaller `model.pth` for deployment.

---

## 🔁 Reproducibility & Scripts

To reproduce training:

```bash
python train.py --dataset path/to/data --epochs 10 --batch-size 256
```

> ⚠️ Training from scratch requires significant compute (GPU recommended).  
> Alternatively, you can use the provided `last.pt` checkpoint and export to `model.pth`.

---

## 🛣️ Roadmap of Future Work

- [ ] Upgrade frontend with richer visualizations (available moves, captured piece display).  
- [ ] Improve AI strength by training longer + on stronger datasets.  
- [ ] Add **MCTS (Monte Carlo Tree Search)** on top of the policy/value net.  

---

## 🙏 Acknowledgements

- [freeCodeCamp](https://www.freecodecamp.org/news/create-a-self-playing-ai-chess-engine-from-scratch/#heading-part-3-how-to-train-the-ai-model) — tutorial that inspired initial architecture.  
- [python-chess](https://python-chess.readthedocs.io) — core library for chess logic + FEN handling.  
- [Stockfish](https://stockfishchess.org) — engine used to generate supervised training data.
- [Lichess](https://database.lichess.org/) - database of games used to train model on

---

## 📜 License

This project is open-source under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

---

## 👤 Author

Developed by **Syed Fadil Uddin**  
- 💼 LinkedIn: [https://www.linkedin.com/in/syed-fadil-uddin/](https://www.linkedin.com/in/syed-fadil-uddin/)  
- 🐙 GitHub: [https://github.com/fadil-uddin](https://github.com/fadil-uddin)  
- ✉️ Email: syedfadiluddin@gmail.com  
