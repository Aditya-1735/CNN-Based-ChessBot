from tensorflow.keras.models import load_model

from_model = load_model(r'C:\Users\Shresth\vscode2\pyfiles\chess-ai\gui\models\1200-elo\from.h5', compile=True)
to_model = load_model(r'C:\Users\Shresth\vscode2\pyfiles\chess-ai\gui\models\1200-elo\to.h5', compile=True)

import pygame 
import chess 

from players import HumanPlayer, AIPlayer
from draw import draw_background, draw_pieces
import globals

pygame.init()

# Game window dimensions
SCREEN_WIDTH = 700
SCREEN_HEIGHT = 600

win = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Chess')

board = chess.Board()

# Initialize different player types
human_player_white = HumanPlayer(colour='white')
human_player_black = HumanPlayer(colour='black')
ai_player_white = AIPlayer(colour='white', from_model=from_model, to_model=to_model)
ai_player_black = AIPlayer(colour='black', from_model=from_model, to_model=to_model)

# Initial setup - Human vs AI
white = human_player_white
black = ai_player_black

fps_clock = pygame.time.Clock()

# Game state variables
run = True 
white_move = True
human_white = True
game_over_countdown = 50

# Game modes
HUMAN_VS_HUMAN = 0
HUMAN_VS_AI = 1
AI_VS_AI = 2
current_mode = HUMAN_VS_AI

def reset():
    board.reset()
    global white_move
    white_move = True 
    
    globals.from_square = None 
    globals.to_square = None

def set_game_mode(mode):
    global white, black, human_white, current_mode
    
    current_mode = mode
    
    if mode == HUMAN_VS_HUMAN:
        white = human_player_white
        black = human_player_black
        human_white = True  # This doesn't matter for human vs human
    elif mode == HUMAN_VS_AI:
        if human_white:
            white = human_player_white
            black = ai_player_black
        else:
            white = ai_player_white
            black = human_player_black
    elif mode == AI_VS_AI:
        white = ai_player_white
        black = ai_player_black
        human_white = True  # This doesn't matter for AI vs AI
    
    reset()

# Add button drawing function
def draw_buttons():
    # Mode selection buttons
    pygame.draw.rect(win, (200, 200, 200), (550, 50, 120, 30))  # Human vs Human
    pygame.draw.rect(win, (200, 200, 200), (550, 90, 120, 30))  # Human vs AI
    pygame.draw.rect(win, (200, 200, 200), (550, 130, 120, 30))  # AI vs AI
    
    # Highlight the selected mode
    if current_mode == HUMAN_VS_HUMAN:
        pygame.draw.rect(win, (150, 150, 220), (550, 50, 120, 30))
    elif current_mode == HUMAN_VS_AI:
        pygame.draw.rect(win, (150, 150, 220), (550, 90, 120, 30))
    elif current_mode == AI_VS_AI:
        pygame.draw.rect(win, (150, 150, 220), (550, 130, 120, 30))
    
    # Swap sides button (only relevant for Human vs AI)
    pygame.draw.rect(win, (200, 200, 200), (625, 200, 50, 60))
    
    # Reset button
    pygame.draw.rect(win, (200, 200, 200), (630, 320, 40, 40))
    
    # Add text for buttons
    font = pygame.font.SysFont('Arial', 12)
    
    human_vs_human_text = font.render('Human vs Human', True, (0, 0, 0))
    human_vs_ai_text = font.render('Human vs AI', True, (0, 0, 0))
    ai_vs_ai_text = font.render('AI vs AI', True, (0, 0, 0))
    swap_text = font.render('Swap', True, (0, 0, 0))
    reset_text = font.render('Reset', True, (0, 0, 0))
    
    win.blit(human_vs_human_text, (560, 60))
    win.blit(human_vs_ai_text, (570, 100))
    win.blit(ai_vs_ai_text, (580, 140))
    win.blit(swap_text, (630, 225))
    win.blit(reset_text, (632, 335))

while run:
    fps_clock.tick(30)

    # Draw game elements
    draw_background(win=win)
    draw_pieces(win=win, fen=board.fen(), human_white=human_white)
    draw_buttons()  # Draw mode selection buttons
    
    pygame.display.update()

    # Handle game over state
    if board.is_game_over():
        # Display result (could be enhanced)
        font = pygame.font.SysFont('Arial', 20)
        if board.is_checkmate():
            winner = "Black" if board.turn else "White"
            result_text = font.render(f"{winner} wins by checkmate!", True, (255, 0, 0))
        else:
            result_text = font.render("Game drawn!", True, (255, 0, 0))
        win.blit(result_text, (240, 280))
        pygame.display.update()
        
        if game_over_countdown > 0:
            game_over_countdown -= 1
        else:
            reset()
            game_over_countdown = 50
        continue

    # AI move logic
    if white_move and (current_mode == AI_VS_AI or (current_mode == HUMAN_VS_AI and not human_white)):
        white.move(board=board, human_white=human_white)
        white_move = not white_move

    if not white_move and (current_mode == AI_VS_AI or (current_mode == HUMAN_VS_AI and human_white)):
        black.move(board=board, human_white=human_white)
        white_move = not white_move

    events = pygame.event.get()

    for event in events:
        if event.type == pygame.QUIT:
            run = False
            pygame.quit()

        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            
            # Mode selection buttons
            if 550 <= x <= 670:
                if 50 <= y <= 80:  # Human vs Human
                    set_game_mode(HUMAN_VS_HUMAN)
                elif 90 <= y <= 120:  # Human vs AI
                    set_game_mode(HUMAN_VS_AI)
                elif 130 <= y <= 160:  # AI vs AI
                    set_game_mode(AI_VS_AI)

            # Swap sides button (only relevant for Human vs AI)
            if 625 <= x <= 675 and 200 <= y <= 260 and current_mode == HUMAN_VS_AI:
                human_white = not human_white
                if human_white:
                    white = human_player_white
                    black = ai_player_black
                else:
                    white = ai_player_white
                    black = human_player_black
                reset()

            # Reset button
            elif 630 <= x <= 670 and 320 <= y <= 360:
                reset()
        
        # Human move logic
        if current_mode != AI_VS_AI:
            if white_move and (current_mode == HUMAN_VS_HUMAN or (current_mode == HUMAN_VS_AI and human_white)):
                if white.move(board=board, event=event, human_white=human_white):
                    white_move = not white_move
            
            elif not white_move and (current_mode == HUMAN_VS_HUMAN or (current_mode == HUMAN_VS_AI and not human_white)):
                if black.move(board=board, event=event, human_white=human_white):
                    white_move = not white_move