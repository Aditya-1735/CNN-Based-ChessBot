# CNN-Minimax Chess Engine

## Overview
I've developed a Python script called `cnn_minimax_chess.py` that integrates a Convolutional Neural Network (CNN) for chess move prediction with a Minimax algorithm enhanced by alpha-beta pruning. This standalone application provides a complete chess-playing experience with an interactive GUI.

## Key Components

### CNNMinimaxEngine Class
- Uses the pre-trained CNN models (`from.h5` and `to.h5`) to predict promising moves
- Implements the Minimax algorithm with alpha-beta pruning
- Combines CNN predictions with Minimax search for optimal move selection
- Evaluates board positions using a sophisticated heuristic function

### ChessUtilities Class
- Provides utility functions for board evaluation
- Implements piece-square tables for positional evaluation
- Handles FEN notation conversions for the CNN input
- Detects endgame positions for specialized evaluation

### ChessGame Class
- Implements a complete GUI chess game using Pygame
- Displays the board, pieces, and legal moves
- Provides an information panel with game status and move history
- Allows switching between different AI modes (CNN Only, Minimax Only, CNN+Minimax)

## Technical Details

### CNN Integration
The script loads pre-trained CNN models that predict the most promising "from" and "to" squares for moves. These models were trained on human chess games and can recognize good move patterns.

### Minimax with Alpha-Beta Pruning
The Minimax algorithm searches through possible move sequences to find the optimal move. Alpha-beta pruning reduces the search space by skipping evaluation of moves that won't affect the final decision.

### Heuristic Evaluation
The board evaluation function considers:
- Material balance (piece values: pawn=100, knight=320, bishop=330, etc.)
- Piece positioning using specialized tables for each piece type
- Different evaluation strategies for middlegame vs endgame
- Mobility (number of legal moves available)
- Game state evaluation (checkmate, stalemate, draws)

### User Interface
The GUI allows players to:
- Play chess against the AI
- Switch sides at any time
- Choose between different AI modes
- View move history
- See highlighted legal moves

## Detailed Technical Implementation

### 1. Sophisticated Heuristic Evaluation Function

The evaluation function combines multiple chess principles into a single numeric score:

```python
def evaluate_board(board):
    """
    Heuristic evaluation function that considers:
    1. Material balance
    2. Piece positioning using piece-square tables
    3. Game state (checkmate, stalemate)
    4. Mobility
    """
    if board.is_checkmate():
        # If current player is checkmated
        return -10000 if board.turn else 10000
    
    if board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves() or board.is_repetition(3):
        return 0  # Draw
    
    endgame = ChessUtilities.is_endgame(board)
    
    # Material and position evaluation
    eval_score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            # Material value
            value = PIECE_VALUES[piece.piece_type]
            
            # Add piece-square table value
            position_value = ChessUtilities.get_piece_square_table_value(
                piece.piece_type, square, piece.color, endgame
            )
            
            # White is positive, black is negative
            modifier = 1 if piece.color else -1
            eval_score += modifier * (value + position_value)
    
    # Mobility evaluation (number of legal moves)
    mobility_bonus = 5  # Points per legal move
    current_legal_moves = len(list(board.legal_moves))
    
    # Store original turn
    original_turn = board.turn
    
    # Switch sides to count opponent's moves
    board.turn = not board.turn
    opponent_legal_moves = len(list(board.legal_moves))
    
    # Restore original turn
    board.turn = original_turn
    
    mobility_score = mobility_bonus * (current_legal_moves - opponent_legal_moves)
    
    # Apply the mobility score
    eval_score += mobility_score if board.turn else -mobility_score
    
    return eval_score
```

This function demonstrates several advanced evaluation concepts:
- Terminal position detection (checkmate, stalemate, etc.)
- Piece-square tables for positional understanding
- Different evaluation strategies for middlegame vs endgame
- Relative mobility evaluation (comparing player's options to opponent's)

### 2. CNN-Minimax Integration

The core innovation is how CNN predictions guide the Minimax search:

```python
def minimax(self, board, depth, alpha, beta, maximizing_player, colour, use_cnn=True):
    """
    Minimax algorithm with alpha-beta pruning.
    Returns the best value and move.
    """
    if depth == 0 or board.is_game_over():
        return ChessUtilities.evaluate_board(board), None
    
    # For the maximizing player (white)
    if maximizing_player:
        max_eval = float('-inf')
        best_move = None
        
        # Here is the key CNN integration - only use CNN at root for efficiency
        if use_cnn and depth == self.depth:
            moves = self.get_cnn_move_predictions(board, colour)
        else:
            # Use all legal moves for deeper levels
            moves = [(move, 0) for move in board.legal_moves]
        
        for move, _ in moves:
            board.push(move)
            eval_score, _ = self.minimax(board, depth - 1, alpha, beta, False, colour, False)
            board.pop()
            
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
                
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # Beta cutoff
                
        return max_eval, best_move
    
    # Similar code for minimizing player...
```

This function shows how:
- The CNN predictions are only used at the root level for efficiency
- At deeper levels, all legal moves are considered to maintain tactical accuracy 
- Alpha-beta pruning speeds up the search by cutting off unpromising branches

### 3. CNN Move Prediction

The CNN move prediction code:

```python
def get_cnn_move_predictions(self, board, colour, top_n=5):
    """
    Get the top N move predictions from the CNN models.
    Returns a list of (move, score) tuples.
    """
    fen = board.fen()
    
    if colour == 'black':
        fen = ChessUtilities.invert_fen(fen=fen)
    
    arr = ChessUtilities.fen_to_matrix(fen=fen)
    arr = arr.reshape((1,) + arr.shape)
    
    # CNN predictions
    from_matrix = self.from_model.predict(arr, verbose=0).reshape((8, 8))
    to_matrix = self.to_model.predict(arr, verbose=0).reshape((8, 8))
    
    if colour == 'black':
        from_matrix = np.flip(from_matrix, axis=0)
        to_matrix = np.flip(to_matrix, axis=0)
    
    # Calculate combined scores for all legal moves
    moves = []
    for move in list(board.legal_moves):
        from_row, from_col, to_row, to_col = ChessUtilities.uci_to_row_col(move.uci())
        score = from_matrix[from_row][from_col] * to_matrix[to_row][to_col]
        moves.append((move, score))
    
    # Sort by CNN score and take top N
    moves.sort(key=lambda x: x[1], reverse=True)
    return moves[:top_n]
```

This function demonstrates:
- How the board state is converted to a format the CNN can understand
- The use of two separate CNN models (from_model and to_model) to predict moves
- How the predictions from both models are combined to score legal moves
- Top-N selection to focus the search on the most promising moves

### 4. Advanced Board Representation

The representation of chess positions for CNN input requires complex transformations:

```python
def fen_to_matrix(fen):
    """Convert a FEN string to a matrix representation for CNN input."""
    fen = fen.split()[0]
    
    piece_dict = {
        'p' : [1,0,0,0,0,0,0,0,0,0,0,0],
        'P' : [0,0,0,0,0,0,1,0,0,0,0,0],
        'n' : [0,1,0,0,0,0,0,0,0,0,0,0],
        'N' : [0,0,0,0,0,0,0,1,0,0,0,0],
        'b' : [0,0,1,0,0,0,0,0,0,0,0,0],
        'B' : [0,0,0,0,0,0,0,0,1,0,0,0],
        'r' : [0,0,0,1,0,0,0,0,0,0,0,0],
        'R' : [0,0,0,0,0,0,0,0,0,1,0,0],
        'q' : [0,0,0,0,1,0,0,0,0,0,0,0],
        'Q' : [0,0,0,0,0,0,0,0,0,0,1,0],
        'k' : [0,0,0,0,0,1,0,0,0,0,0,0],
        'K' : [0,0,0,0,0,0,0,0,0,0,0,1],
        '.' : [0,0,0,0,0,0,0,0,0,0,0,0],
    }

    row_arr = []
    rows = fen.split('/')
    for row in rows:
        arr = []
        for ch in str(row):
            if ch.isdigit():
                for _ in range(int(ch)):
                    arr.append(piece_dict['.'])
            else:
                arr.append(piece_dict[ch])
        row_arr.append(arr)

    mat = np.array(row_arr)
    return mat
```

This shows:
- How FEN chess notation is converted to a 12-channel input for the CNN
- Each piece type (pawn, knight, etc.) for each color gets its own channel
- This one-hot encoding preserves the spatial relationship of pieces on the board

### 5. Integration of Game Modes

The solution implements different AI modes through a flexible architecture:

```python
def find_best_move(self, board, colour, use_cnn=True):
    """Find the best move for the given board position."""
    start_time = time.time()
    
    is_maximizing = colour == 'white'
    
    # Get the best move using minimax with configurable CNN usage
    _, best_move = self.minimax(
        board, 
        self.depth, 
        float('-inf'), 
        float('inf'), 
        is_maximizing,
        colour,
        use_cnn  # Toggle CNN on/off based on mode selection
    )
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # If no best move found (shouldn't happen unless game over), return first legal move
    if best_move is None and not board.is_game_over():
        best_move = list(board.legal_moves)[0]
        
    return best_move, elapsed
```

This demonstrates:
- How the AI engine can be dynamically adjusted between CNN-guided mode and standard Minimax 
- Flexible control over the specific behaviors of the Minimax algorithm
- Performance timing to measure computational efficiency

## How It Works
1. For each position, the CNN models predict the most promising moves
2. These moves are then evaluated using the Minimax algorithm
3. The evaluation considers material, position, and other chess principles
4. The AI selects the move with the highest evaluation score

This approach combines the strengths of neural networks (pattern recognition from human games) with traditional search algorithms (tactical calculation), creating a more balanced and powerful chess AI.