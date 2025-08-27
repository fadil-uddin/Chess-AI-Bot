"""
ChessAIBot_ExportModel.py

This script loads the full training checkpoint (last.pt),
extracts only the trained model weights, and saves them
into a lightweight .pth file for inference.

Author: [Syed Fadil Uddin]
"""

import torch
import argparse
from ChessAIBot_Model import PolicyValueNet  # <-- make sure this matches your project structure

def export_checkpoint(input_file, output_file):
    # Load the checkpoint
    checkpoint = torch.load(input_file, map_location="cpu")

    # Initialize model with the same architecture
    model = PolicyValueNet()

    # Load model weights only
    model.load_state_dict(checkpoint["model"])

    # Save only the model weights (state_dict), nothing else
    torch.save(model.state_dict(), output_file)

    print(f"✅ Exported model weights saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input checkpoint (e.g. last.pt)")
    parser.add_argument("--output", type=str, required=True, help="Path to output weights file (e.g. model.pth)")
    args = parser.parse_args()

    export_checkpoint(args.input, args.output)


