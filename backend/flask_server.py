import sys
import os
sys.path.append(os.path.abspath(r"C:\Users\Shresth\vscode2\pyfiles\chess-ai\gui"))  # Adjust the path as needed

from cnn_minimax_chess import CNNMinimaxEngine
import chess

from flask import Flask, request, jsonify

from flask_cors import CORS  # Import the Flask-CORS extension

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the chess engine with default parameters
# You can adjust these parameters as needed
engine = CNNMinimaxEngine(elo="1200", depth=3)

@app.route('/api/move', methods=['POST'])
def make_move():
    """
    API endpoint to get the AI's next move based on the current board state.
    
    Expected JSON payload:
    {
        "fen": "current board position in FEN notation",
        "player": "white" or "black",
        "mode": "cnn_only", "minimax_only", or "cnn_minimax"
    }
    
    Returns:
    {
        "move": "move in UCI notation (e.g., 'e2e4')",
        "time_taken": elapsed time in seconds
    }
    """
    try:
        # Get the request data
        data = request.json
        print(data)
        
        if not data or 'fen' not in data or 'player' not in data or 'mode' not in data:
            return jsonify({"error": "Missing required fields: fen, player, or mode"}), 400
            
        fen = data['fen']
        player = data['player']
        mode = data['mode']
        
        # Validate the mode
        if mode not in ['cnn_only', 'minimax_only', 'cnn_minimax']:
            return jsonify({"error": "Invalid mode. Must be 'cnn_only', 'minimax_only', or 'cnn_minimax'"}), 400
            
        # Create a chess board from the FEN string
        board = chess.Board(fen)
        
        # Check if the game is over
        if board.is_game_over():
            return jsonify({"error": "Game is over", "result": board.result()}), 400
            
        # Select the appropriate move function based on the mode
        if mode == 'cnn_only':
            # For CNN-only, get the top move prediction and use it directly
            cnn_moves = engine.get_cnn_move_predictions(board, player, top_n=1)
            if not cnn_moves:
                return jsonify({"error": "No valid moves found"}), 400
            best_move, score = cnn_moves[0]
            time_taken = 0  # CNN prediction is typically very fast
        elif mode == 'minimax_only':
            # For Minimax-only, use find_best_move with use_cnn=False
            best_move, time_taken = engine.find_best_move(board, player, use_cnn=False)
        else:  # cnn_minimax
            # For CNN+Minimax, use find_best_move with use_cnn=True
            best_move, time_taken = engine.find_best_move(board, player, use_cnn=True)
            
        # Return the move in UCI notation
        print(best_move.uci())
        return jsonify({
            "move": best_move.uci(),
            "time_taken": time_taken
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Add a simple health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

