import pygame
import numpy as np

from globals import board_colour, square_size
import globals

# Load GIF chess piece images
wp = pygame.image.load(r'gui\images\white_pawn.gif')
bp = pygame.image.load(r'gui\images\black_pawn.gif')
wn = pygame.image.load(r'gui\images\white_knight.gif')
bn = pygame.image.load(r'gui\images\black_knight.gif')
wb = pygame.image.load(r'gui\images\white_bishop.gif')
bb = pygame.image.load(r'gui\images\black_bishop.gif')
wr = pygame.image.load(r'gui\images\white_rook.gif')
br = pygame.image.load(r'gui\images\black_rook.gif')
wq = pygame.image.load(r'gui\images\white_queen.gif')
bq = pygame.image.load(r'gui\images\black_queen.gif')
wk = pygame.image.load(r'gui\images\white_king.gif')
bk = pygame.image.load(r'gui\images\black_king.gif')

# Scale the GIF pieces to fit the square size
piece_size = int(square_size * 0.8)  # Make pieces 80% of the square size
x_offset = (square_size - piece_size) // 2
y_offset = (square_size - piece_size) // 2

wp = pygame.transform.scale(wp, (piece_size, piece_size))
bp = pygame.transform.scale(bp, (piece_size, piece_size))
wn = pygame.transform.scale(wn, (piece_size, piece_size))
bn = pygame.transform.scale(bn, (piece_size, piece_size))
wb = pygame.transform.scale(wb, (piece_size, piece_size))
bb = pygame.transform.scale(bb, (piece_size, piece_size))
wr = pygame.transform.scale(wr, (piece_size, piece_size))
br = pygame.transform.scale(br, (piece_size, piece_size))
wq = pygame.transform.scale(wq, (piece_size, piece_size))
bq = pygame.transform.scale(bq, (piece_size, piece_size))
wk = pygame.transform.scale(wk, (piece_size, piece_size))
bk = pygame.transform.scale(bk, (piece_size, piece_size))

switch = pygame.image.load(r'C:\Users\Shresth\vscode2\pyfiles\chess-ai\gui\images/switch.png')
switch = pygame.transform.scale(switch, (50, 60))

restart = pygame.image.load(r'C:\Users\Shresth\vscode2\pyfiles\chess-ai\gui\images/restart.png')
restart = pygame.transform.scale(restart, (40, 40))

FROM_COLOUR = (45,22,178)
TO_COLOUR = (223, 223, 12)

def draw_background(win):
    win.fill((118, 150, 86))

    for x in range(8):
        for y in range(0, 8):
            if (x % 2 == 1 and y % 2 == 0) or (x % 2 == 0 and y % 2 == 1):
                pygame.draw.rect(win, board_colour, (x * square_size, y * square_size, square_size, square_size))
    
    if globals.from_square:
        pygame.draw.rect(win, FROM_COLOUR, (globals.from_square[0] * square_size, globals.from_square[1] * square_size, square_size, square_size))

    if globals.to_square:
        pygame.draw.rect(win, TO_COLOUR, (globals.to_square[0] * square_size, globals.to_square[1] * square_size, square_size, square_size))

    win.blit(switch, (625, 200))
    win.blit(restart, (630, 320))

def draw_pieces(win, fen, human_white):
    def fen_to_array(fen):
        fen = fen.split()[0]

        arr = []

        rows = fen.split('/')
        for row in rows:
            row_arr = []
            for ch in str(row):
                if ch.isdigit():
                    for _ in range(int(ch)):
                        row_arr.append('.')
                else:
                    row_arr.append(ch)
            arr.append(row_arr)

        return arr
     
    arr = fen_to_array(fen=fen)

    piece_to_variable = {
        'p': bp,
        'n': bn,
        'b': bb,
        'r': br,
        'q': bq,
        'k': bk,
        'P': wp,
        'N': wn,
        'B': wb,
        'R': wr,
        'Q': wq,
        'K': wk, 
    }

    if not human_white:
        arr = np.array(arr)
        arr = np.flip(arr, axis=[0, 1])

    for x in range(8):
        for y in range(8):
            
            if arr[y][x] == '.':
                continue 
            
            piece = piece_to_variable[arr[y][x]]

            # Center the piece within the square using the offsets
            win.blit(piece, (x * square_size + x_offset, y * square_size + y_offset))