import numpy as np
import chess
import time
import argparse
from tensorflow.keras.models import load_model # type: ignore
import pygame
import os

# Set up paths relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GUI_DIR = SCRIPT_DIR  # Since the script is already in the gui directory
MODEL_DIRS = {
    "700": os.path.join(GUI_DIR, "models", "700-elo"),
    "1100": os.path.join(GUI_DIR, "models", "1100-elo"),
    "1200": os.path.join(GUI_DIR, "models", "1200-elo")
}

# Piece values for heuristic evaluation (in centipawns)
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# Piece-square tables for positional evaluation
PAWN_TABLE = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [10, 10, 20, 30, 30, 20, 10, 10],
    [5, 5, 10, 25, 25, 10, 5, 5],
    [0, 0, 0, 20, 20, 0, 0, 0],
    [5, -5, -10, 0, 0, -10, -5, 5],
    [5, 10, 10, -20, -20, 10, 10, 5],
    [0, 0, 0, 0, 0, 0, 0, 0]
])

KNIGHT_TABLE = np.array([
    [-50, -40, -30, -30, -30, -30, -40, -50],
    [-40, -20, 0, 0, 0, 0, -20, -40],
    [-30, 0, 10, 15, 15, 10, 0, -30],
    [-30, 5, 15, 20, 20, 15, 5, -30],
    [-30, 0, 15, 20, 20, 15, 0, -30],
    [-30, 5, 10, 15, 15, 10, 5, -30],
    [-40, -20, 0, 5, 5, 0, -20, -40],
    [-50, -40, -30, -30, -30, -30, -40, -50]
])

BISHOP_TABLE = np.array([
    [-20, -10, -10, -10, -10, -10, -10, -20],
    [-10, 0, 0, 0, 0, 0, 0, -10],
    [-10, 0, 10, 10, 10, 10, 0, -10],
    [-10, 5, 5, 10, 10, 5, 5, -10],
    [-10, 0, 5, 10, 10, 5, 0, -10],
    [-10, 10, 10, 10, 10, 10, 10, -10],
    [-10, 5, 0, 0, 0, 0, 5, -10],
    [-20, -10, -10, -10, -10, -10, -10, -20]
])

ROOK_TABLE = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [5, 10, 10, 10, 10, 10, 10, 5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [0, 0, 0, 5, 5, 0, 0, 0]
])

QUEEN_TABLE = np.array([
    [-20, -10, -10, -5, -5, -10, -10, -20],
    [-10, 0, 0, 0, 0, 0, 0, -10],
    [-10, 0, 5, 5, 5, 5, 0, -10],
    [-5, 0, 5, 5, 5, 5, 0, -5],
    [0, 0, 5, 5, 5, 5, 0, -5],
    [-10, 5, 5, 5, 5, 5, 0, -10],
    [-10, 0, 5, 0, 0, 0, 0, -10],
    [-20, -10, -10, -5, -5, -10, -10, -20]
])

KING_MIDDLE_GAME_TABLE = np.array([
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-20, -30, -30, -40, -40, -30, -30, -20],
    [-10, -20, -20, -20, -20, -20, -20, -10],
    [20, 20, 0, 0, 0, 0, 20, 20],
    [20, 30, 10, 0, 0, 10, 30, 20]
])

KING_END_GAME_TABLE = np.array([
    [-50, -40, -30, -20, -20, -30, -40, -50],
    [-30, -20, -10, 0, 0, -10, -20, -30],
    [-30, -10, 20, 30, 30, 20, -10, -30],
    [-30, -10, 30, 40, 40, 30, -10, -30],
    [-30, -10, 30, 40, 40, 30, -10, -30],
    [-30, -10, 20, 30, 30, 20, -10, -30],
    [-30, -30, 0, 0, 0, 0, -30, -30],
    [-50, -30, -30, -30, -30, -30, -30, -50]
])

class ChessUtilities:
    @staticmethod
    def is_endgame(board):
        """Determine if the current position is an endgame."""
        queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
        total_pieces = len(list(board.pieces(chess.KNIGHT, chess.WHITE))) + len(list(board.pieces(chess.KNIGHT, chess.BLACK))) + \
                    len(list(board.pieces(chess.BISHOP, chess.WHITE))) + len(list(board.pieces(chess.BISHOP, chess.BLACK))) + \
                    len(list(board.pieces(chess.ROOK, chess.WHITE))) + len(list(board.pieces(chess.ROOK, chess.BLACK)))
        
        return queens == 0 or total_pieces <= 6

    @staticmethod
    def get_piece_square_table_value(piece_type, square, is_white, endgame=False):
        """Get the position value for a piece from its appropriate piece-square table."""
        row = 7 - chess.square_rank(square) if is_white else chess.square_rank(square)
        col = chess.square_file(square)
        
        if piece_type == chess.PAWN:
            return PAWN_TABLE[row][col]
        elif piece_type == chess.KNIGHT:
            return KNIGHT_TABLE[row][col]
        elif piece_type == chess.BISHOP:
            return BISHOP_TABLE[row][col]
        elif piece_type == chess.ROOK:
            return ROOK_TABLE[row][col]
        elif piece_type == chess.QUEEN:
            return QUEEN_TABLE[row][col]
        elif piece_type == chess.KING:
            if endgame:
                return KING_END_GAME_TABLE[row][col]
            else:
                return KING_MIDDLE_GAME_TABLE[row][col]
        return 0

    @staticmethod
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
        
        # Additional positional bonuses/penalties
        # Mobility (number of legal moves)
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

    @staticmethod
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

    @staticmethod
    def invert_fen(fen):
        """Invert the FEN representation for black's perspective."""
        fen = fen.split()[0]
        rows = fen.split('/')
        rows.reverse()
        for i in range(8):
            rows[i] = rows[i].swapcase()
        return '/'.join(rows)

    @staticmethod
    def uci_to_row_col(uci):
        """Convert UCI move notation to row-column coordinates."""
        sq_from = uci[: 2]
        sq_to = uci[2 :]

        def parse_square(sq):
            col = ord(sq[0]) - ord('a')
            row = 7 - (int(sq[1]) - 1)
            return row, col

        return parse_square(sq_from) + parse_square(sq_to)

class CNNMinimaxEngine:
    def __init__(self, elo="1200", depth=3):
        self.depth = depth
        self.from_model = load_model(os.path.join(MODEL_DIRS[elo], "from.h5"), compile=True)
        self.to_model = load_model(os.path.join(MODEL_DIRS[elo], "to.h5"), compile=True)
        
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
        
        from_matrix = self.from_model.predict(arr, verbose=0).reshape((8, 8))
        to_matrix = self.to_model.predict(arr, verbose=0).reshape((8, 8))
        
        if colour == 'black':
            from_matrix = np.flip(from_matrix, axis=0)
            to_matrix = np.flip(to_matrix, axis=0)
        
        moves = []
        for move in list(board.legal_moves):
            from_row, from_col, to_row, to_col = ChessUtilities.uci_to_row_col(move.uci())
            score = from_matrix[from_row][from_col] * to_matrix[to_row][to_col]
            moves.append((move, score))
        
        # Sort by CNN score and take top N
        moves.sort(key=lambda x: x[1], reverse=True)
        return moves[:top_n]

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
            
            # Get top moves from CNN or all legal moves
            if use_cnn and depth == self.depth:  # Only use CNN at the root for efficiency
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
        
        # For the minimizing player (black)
        else:
            min_eval = float('inf')
            best_move = None
            
            # Get top moves from CNN or all legal moves
            if use_cnn and depth == self.depth:  # Only use CNN at the root for efficiency
                moves = self.get_cnn_move_predictions(board, colour)
            else:
                # Use all legal moves for deeper levels
                moves = [(move, 0) for move in board.legal_moves]
                
            for move, _ in moves:
                board.push(move)
                eval_score, _ = self.minimax(board, depth - 1, alpha, beta, True, colour, False)
                board.pop()
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                    
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cutoff
                    
            return min_eval, best_move

    def find_best_move(self, board, colour, use_cnn=True):
        """Find the best move for the given board position."""
        start_time = time.time()
        
        is_maximizing = colour == 'white'
        
        # Get the best move using minimax
        _, best_move = self.minimax(
            board, 
            self.depth, 
            float('-inf'), 
            float('inf'), 
            is_maximizing,
            colour,
            use_cnn
        )
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # If no best move found (shouldn't happen unless game over), return first legal move
        if best_move is None and not board.is_game_over():
            best_move = list(board.legal_moves)[0]
            
        return best_move, elapsed

class ChessGame:
    def __init__(self, elo="1200", depth=3, square_size=75):
        pygame.init()
        self.square_size = square_size
        self.board_size = 8 * square_size
        self.info_width = 300
        self.screen_width = self.board_size + self.info_width
        self.screen_height = self.board_size
        
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('Chess AI with CNN-Minimax')
        
        self.board = chess.Board()
        self.engine = CNNMinimaxEngine(elo=elo, depth=depth)
        
        # Load piece images
        self.images = {}
        self.load_images()
        
        # Game state
        self.player_white = True
        self.ai_thinking = False
        self.selected_square = None
        self.game_over = False
        self.last_move = None
        self.move_history = []
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.LIGHT_SQUARE = (240, 217, 181)
        self.DARK_SQUARE = (181, 136, 99)
        self.HIGHLIGHT = (124, 252, 0, 128)  # Light green with some transparency
        self.SELECTED = (255, 255, 0, 150)   # Yellow with some transparency
        self.LAST_MOVE = (186, 202, 43, 150) # Light olive
        
        # Fonts
        self.font = pygame.font.SysFont('Arial', 24)
        self.small_font = pygame.font.SysFont('Arial', 18)
        
        # Buttons
        self.buttons = [
            {'text': 'New Game', 'rect': pygame.Rect(self.board_size + 20, 50, 120, 40)},
            {'text': 'Switch Sides', 'rect': pygame.Rect(self.board_size + 150, 50, 120, 40)},
            {'text': 'CNN Only', 'rect': pygame.Rect(self.board_size + 20, 100, 120, 40)},
            {'text': 'Minimax Only', 'rect': pygame.Rect(self.board_size + 150, 100, 120, 40)},
            {'text': 'CNN+Minimax', 'rect': pygame.Rect(self.board_size + 85, 150, 120, 40)},
        ]
        self.use_cnn = True
        self.current_mode = "CNN+Minimax"
    
    def load_images(self):
        """Load chess piece images."""
        # Map chess piece symbols to the correct image filenames
        piece_filenames = {
            'P': 'pawn',
            'R': 'rook',
            'N': 'knight',
            'B': 'bishop',
            'Q': 'queen',
            'K': 'king'
        }
        
        for piece, filename in piece_filenames.items():
            # Load white pieces
            self.images[piece] = pygame.image.load(os.path.join(GUI_DIR, "images", f"white_{filename}.gif"))
            self.images[piece] = pygame.transform.scale(self.images[piece], (self.square_size, self.square_size))
            
            # Load black pieces
            self.images[piece.lower()] = pygame.image.load(os.path.join(GUI_DIR, "images", f"black_{filename}.gif"))
            self.images[piece.lower()] = pygame.transform.scale(self.images[piece.lower()], (self.square_size, self.square_size))
    
    def draw_board(self):
        """Draw the chess board and pieces."""
        # Draw squares
        for row in range(8):
            for col in range(8):
                color = self.LIGHT_SQUARE if (row + col) % 2 == 0 else self.DARK_SQUARE
                pygame.draw.rect(self.screen, color, 
                                (col * self.square_size, row * self.square_size, 
                                self.square_size, self.square_size))
                
                # Draw rank and file labels
                if col == 0:  # Ranks on the left
                    rank_label = self.small_font.render(str(8 - row), True, self.BLACK if color == self.LIGHT_SQUARE else self.WHITE)
                    self.screen.blit(rank_label, (5, row * self.square_size + 5))
                
                if row == 7:  # Files on the bottom
                    file_label = self.small_font.render(chr(97 + col), True, self.BLACK if color == self.LIGHT_SQUARE else self.WHITE)
                    self.screen.blit(file_label, (col * self.square_size + self.square_size - 15, self.board_size - 20))
        
        # Highlight last move
        if self.last_move:
            from_square = chess.parse_square(self.last_move[:2])
            to_square = chess.parse_square(self.last_move[2:4])
            
            # Create a transparent surface for the highlight
            highlight_surface = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
            pygame.draw.rect(highlight_surface, self.LAST_MOVE, (0, 0, self.square_size, self.square_size))
            
            # Highlight from square
            from_col = chess.square_file(from_square)
            from_row = 7 - chess.square_rank(from_square)
            self.screen.blit(highlight_surface, (from_col * self.square_size, from_row * self.square_size))
            
            # Highlight to square
            to_col = chess.square_file(to_square)
            to_row = 7 - chess.square_rank(to_square)
            self.screen.blit(highlight_surface, (to_col * self.square_size, to_row * self.square_size))
        
        # Highlight selected square
        if self.selected_square:
            square = chess.parse_square(self.selected_square)
            col = chess.square_file(square)
            row = 7 - chess.square_rank(square)
            
            # Create a transparent surface for the highlight
            highlight_surface = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
            pygame.draw.rect(highlight_surface, self.SELECTED, (0, 0, self.square_size, self.square_size))
            
            self.screen.blit(highlight_surface, (col * self.square_size, row * self.square_size))
            
            # Highlight legal moves from selected square
            for move in self.board.legal_moves:
                if move.from_square == square:
                    to_col = chess.square_file(move.to_square)
                    to_row = 7 - chess.square_rank(move.to_square)
                    
                    # Create a transparent surface for the highlight
                    move_surface = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
                    pygame.draw.rect(move_surface, self.HIGHLIGHT, (0, 0, self.square_size, self.square_size))
                    
                    self.screen.blit(move_surface, (to_col * self.square_size, to_row * self.square_size))
        
        # Draw pieces
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                # Convert chess.Square to row, col
                col = chess.square_file(square)
                row = 7 - chess.square_rank(square)  # Invert row since chess uses 1-8 from bottom
                
                piece_symbol = piece.symbol()
                self.screen.blit(self.images[piece_symbol], 
                                (col * self.square_size, row * self.square_size))
    
    def draw_info_panel(self):
        """Draw information panel on the right side."""
        # Draw background
        pygame.draw.rect(self.screen, self.WHITE, 
                        (self.board_size, 0, self.info_width, self.screen_height))
        
        # Draw border
        pygame.draw.line(self.screen, self.BLACK, 
                        (self.board_size, 0), (self.board_size, self.screen_height), 2)
        
        # Draw game status
        status_text = "White's turn" if self.board.turn else "Black's turn"
        if self.board.is_checkmate():
            status_text = "Checkmate! " + ("Black" if self.board.turn else "White") + " wins!"
        elif self.board.is_stalemate():
            status_text = "Stalemate!"
        elif self.board.is_insufficient_material():
            status_text = "Draw by insufficient material!"
        
        status_surface = self.font.render(status_text, True, self.BLACK)
        self.screen.blit(status_surface, (self.board_size + 20, 10))
        
        # Draw current mode
        mode_surface = self.font.render(f"Mode: {self.current_mode}", True, self.BLACK)
        self.screen.blit(mode_surface, (self.board_size + 20, 200))
        
        # Draw buttons
        for button in self.buttons:
            pygame.draw.rect(self.screen, (200, 200, 200), button['rect'])
            pygame.draw.rect(self.screen, self.BLACK, button['rect'], 2)
            
            button_text = self.small_font.render(button['text'], True, self.BLACK)
            text_rect = button_text.get_rect(center=button['rect'].center)
            self.screen.blit(button_text, text_rect)
        
        # Display move history
        history_title = self.font.render("Move History", True, self.BLACK)
        self.screen.blit(history_title, (self.board_size + 20, 250))
        
        # Display last 10 moves
        for i, move in enumerate(self.move_history[-10:]):
            move_num = len(self.move_history) - 10 + i + 1
            move_text = f"{move_num}. {move}"
            move_surface = self.small_font.render(move_text, True, self.BLACK)
            self.screen.blit(move_surface, (self.board_size + 20, 280 + i * 20))
    
    def handle_mouse_click(self, pos):
        """Handle mouse click events."""
        # Check if clicked on a button
        for button in self.buttons:
            if button['rect'].collidepoint(pos):
                if button['text'] == 'New Game':
                    self.board = chess.Board()
                    self.selected_square = None
                    self.game_over = False
                    self.last_move = None
                    self.move_history = []
                elif button['text'] == 'Switch Sides':
                    self.player_white = not self.player_white
                    if self.board.turn != self.player_white:
                        self.ai_thinking = True  # Trigger AI move if it's AI's turn after switch
                elif button['text'] == 'CNN Only':
                    self.use_cnn = True
                    self.current_mode = "CNN Only"
                elif button['text'] == 'Minimax Only':
                    self.use_cnn = False
                    self.current_mode = "Minimax Only"
                elif button['text'] == 'CNN+Minimax':
                    self.use_cnn = True
                    self.current_mode = "CNN+Minimax"
                return
        
        # If game over or AI's turn, ignore board clicks
        if self.game_over or (self.board.turn and not self.player_white) or (not self.board.turn and self.player_white):
            return
        
        # Handle click on board
        if pos[0] < self.board_size and pos[1] < self.board_size:
            col = pos[0] // self.square_size
            row = pos[1] // self.square_size
            
            # Convert to chess square notation
            file_letter = chr(97 + col)  # 'a' through 'h'
            rank_number = 8 - row       # 1 through 8
            square_name = file_letter + str(rank_number)
            
            if self.selected_square is None:
                # Check if there's a piece on the square and it's the player's turn
                square = chess.parse_square(square_name)
                piece = self.board.piece_at(square)
                if piece and ((piece.color and self.player_white) or (not piece.color and not self.player_white)):
                    self.selected_square = square_name
            else:
                # Attempt to make a move
                move_uci = self.selected_square + square_name
                try:
                    move = chess.Move.from_uci(move_uci)
                    if move in self.board.legal_moves:
                        # Check for promotion
                        if self.board.piece_at(chess.parse_square(self.selected_square)).piece_type == chess.PAWN:
                            from_rank = int(self.selected_square[1])
                            to_rank = int(square_name[1])
                            if (from_rank == 7 and to_rank == 8) or (from_rank == 2 and to_rank == 1):
                                move.promotion = chess.QUEEN
                        
                        # Make the move
                        self.board.push(move)
                        self.last_move = move_uci
                        self.move_history.append(move_uci)
                        self.ai_thinking = True  # Now it's AI's turn
                    else:
                        # If move is not legal, try selecting a new piece
                        square = chess.parse_square(square_name)
                        piece = self.board.piece_at(square)
                        if piece and ((piece.color and self.player_white) or (not piece.color and not self.player_white)):
                            self.selected_square = square_name
                        else:
                            self.selected_square = None
                except ValueError:
                    # Invalid move format, reset selection
                    self.selected_square = None
    
    def make_ai_move(self):
        """Make a move for the AI."""
        if self.board.is_game_over():
            self.game_over = True
            return
        
        # Determine AI's color
        ai_color = 'black' if self.player_white else 'white'
        
        # Find best move
        best_move, elapsed = self.engine.find_best_move(self.board, ai_color, self.use_cnn)
        
        if best_move:
            # Apply the move
            move_uci = best_move.uci()
            self.board.push(best_move)
            self.last_move = move_uci
            self.move_history.append(move_uci)
            print(f"AI Move: {move_uci} (calculated in {elapsed:.2f} seconds)")
        
        # Check if game is over
        if self.board.is_game_over():
            self.game_over = True
    
    def run(self):
        """Main game loop."""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_mouse_click(event.pos)
            
            # Make AI move if it's AI's turn
            if self.ai_thinking and not self.game_over:
                self.ai_thinking = False  # Reset flag
                self.make_ai_move()
            
            # If player is not the current turn, AI should move
            if ((self.board.turn and not self.player_white) or (not self.board.turn and self.player_white)) and not self.game_over and not self.ai_thinking:
                self.ai_thinking = True
            
            # Draw everything
            self.screen.fill(self.WHITE)
            self.draw_board()
            self.draw_info_panel()
            
            pygame.display.flip()
            clock.tick(30)
        
        pygame.quit()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Chess AI with CNN and Minimax')
    parser.add_argument('--elo', choices=['700', '1100', '1200'], default='1200', help='ELO rating of the CNN model to use')
    parser.add_argument('--depth', type=int, default=3, help='Minimax search depth')
    
    args = parser.parse_args()
    
    game = ChessGame(elo=args.elo, depth=args.depth)
    game.run()

if __name__ == "__main__":
    main()