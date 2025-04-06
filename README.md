# CNN-Based-ChessBot

## Course Information
**Course Code:** CS2203 – Artificial Intelligence  
**Project Title:** MKAI – Move Prediction using CNNs in Chess

## Team Members
| Name | Roll Number |
|------|------------|
| Shresth Kasyap | 2301AI22 |
| Harsh Raj | 2301AI47 |
| Sasmit Shashwat | 2301AI20 |
| Arush Singh Kiron | 2301AI03 |

## Project Overview
This project implements a chess-playing AI that leverages deep convolutional neural networks to predict optimal moves from given board positions. Our approach decomposes move prediction into two parallel subtasks:

1. **Move-from prediction**: Identifying the source square from which a piece should be moved
2. **Move-to prediction**: Determining the destination square for the selected piece

These components are trained independently through supervised learning on preprocessed chess positions in FEN (Forsyth-Edwards Notation) format.

The final move selection combines both models by computing a joint probability score:
```
score(move) = from_model[from_square] × to_model[to_square]
```

## Technical Features
- **Architecture**: Dual-headed CNN design with specialized prediction paths
- **Data Pipeline**: Complete PGN-to-FEN conversion with automated label generation
- **Representation**: One-hot encoded board states capturing piece types and positions
- **Interface**: Interactive GUI for human-vs-AI gameplay
- **Reproducibility**: Comprehensive training and evaluation scripts

## Implementation Guide

### Requirements
The system requires Python 3.8 or newer. Install all dependencies with:
```bash
python -m pip install -r requirements.txt
```

### Launch the Interface
```bash
cd gui
python main.py
```

## Dataset Processing
Our data originates from the [Lichess public PGN database](https://database.lichess.org) and undergoes the following processing steps:

1. **Position Extraction**: Converting PGN game records to FEN positions after each move
2. **Move Labeling**: Analyzing consecutive positions to determine moves made
3. **Perspective Normalization**: Converting all positions to white's perspective (mirroring black's positions)
4. **Tensor Encoding**: Transforming each position into an 8×8×12 tensor (8×8 board with 12 channels for piece types)
5. **Target Creation**: Generating binary matrices for source and target squares

The processed dataset contains entries in the format:
```
FEN  from_square_index  to_square_index
```

Core preprocessing logic is implemented in `util.py` and `train.py`.

## Neural Network Architecture
The model architecture (defined in `model.py`) consists of:

### Network Structure
- **Input Layer**: 8×8×12 tensor representing the chess board state
- **Feature Extraction**: 6 convolutional blocks with batch normalization
- **Skip Connections**: Residual pathways to preserve spatial information
- **Refinement**: Spatially-applied dense layers for feature integration
- **Output**: 8×8×1 probability grid with softmax activation

Both "from" and "to" models share this architecture but are trained independently.

### Technical Specifications
```
Input: (8, 8, 12)
→ Conv2D (32 filters) → BatchNorm → ReLU
→ Conv2D (64 filters) → BatchNorm → ReLU
→ Conv2D (256 filters) → BatchNorm → ReLU
→ Residual concatenation
→ 3× Conv2D (256 filters)
→ Dense (256) → Dense (64) → Dense (1)
→ Softmax activation (8×8 grid)
```

- Parameters: ~2.8 million per model
- Loss Function: Categorical Cross-entropy
- Optimization: Adam optimizer

## Training Process
The training pipeline is managed by `train.py` using TensorFlow's data API.

### Hyperparameters
- Batch Size: 1024
- Training Epochs: 10
- Steps per Epoch: 1000
- Validation Proportion: 10%
- Checkpoint Pattern: `gm_from/model{epoch}.h5`

### Training Commands
To train either model component (controlled by `TRAIN_MOVE_FROM` flag):
```bash
python train.py
```

To save model weights separately:
```bash
python save_weights.py
```

Training progress can be monitored through TensorBoard at `logs/callback/`.

## Inference and Evaluation
Use `test.py` to evaluate the model with custom positions:

```python
fen = 'rnbqkbnr/pppp1ppp/B7/4p3/4P3/8/PPPP1PPP/RNBQK1NR b KQkq - 1 2'
```

The inference pipeline:
1. Detects active player (white/black)
2. Normalizes board representation
3. Loads appropriate model weights
4. Computes square probabilities
5. Outputs prediction matrices and highest-probability squares

## Interactive Interface
The GUI is built with `pygame` and `python-chess`, providing:
- Intuitive drag-and-drop piece movement
- Visual representation of model predictions
- Legal move validation and enforcement
- Game state management (reset, custom positions)

To launch the interface:
```bash
cd gui
python main.py
```
