import pygame as p

import gameState
from PawnPiece import PawnPiece

WIDTH = 600
HEIGHT = 600
DIMENSION = 15
SQ_SIZE = HEIGHT / (DIMENSION-1)
MAX_FPS = 15
IMAGES = {}

# load images 1 time. Reloading images is taxing

'''
Initialize a global dictionary of images. Called 1 time. 
'''


def load_images():
    # access image by IMAGES['wp']
    pieces = ['bC', 'wC', 'wp', 'wR', 'wN', 'wB', 'wk', 'wQ', 'bp', 'bR', 'bN', 'bB', 'bK', 'bQ']
    for piece in pieces:
        IMAGES[piece] = p.transform.scale(
            p.image.load("C:/Users/VMShk/Desktop/CS school/pythonChess/resources/" + piece + ".png"),
            (SQ_SIZE, SQ_SIZE))


def main():
    p.init()
    screen = p.display.set_mode((WIDTH, HEIGHT))
    clock = p.time.Clock()
    screen.fill(p.Color("white"))
    gs = gameState.GameState()

    load_images()
    running = True
    sqSelected = ()
    player_turn = 0




    while running:

        for e in p.event.get():
            if e.type == p.QUIT:
                running = False

            elif e.type == p.MOUSEBUTTONDOWN:
                location = p.mouse.get_pos()  # tracks mouse xy
                print(location)
                col = int((location[0] - SQ_SIZE * .5) // SQ_SIZE)
                row = int((location[1] - SQ_SIZE * .5) // SQ_SIZE)


                if gs.matrix[row][col] == "empty" and player_turn == 0:
                    gs.matrix[row][col] = PawnPiece(row, col, 'black')
                    player_turn = player_turn + 1
                    player_turn = player_turn % 2

                if gs.matrix[row][col] == "empty" and player_turn == 1:
                    gs.matrix[row][col] = PawnPiece(row, col, 'white')
                    player_turn = player_turn + 1
                    player_turn = player_turn % 2







        drawGameState(screen, gs, sqSelected)
        clock.tick(MAX_FPS)
        black = (0, 0, 0)
        myFont = p.font.SysFont("Times New Roman", 26)
        player = ''
        if player_turn == 0:
            player = 'black'
        if player_turn == 1:
            player = 'white'

        player_display = myFont.render(player, 1, black)
        screen.blit(player_display, (200, 220))
        p.display.flip()


def drawGameState(screen, gs, sqSelected):
    drawBoard(screen)
    drawPieces(screen, gs.matrix)

    highlight = gs.get_highlight_switch()

    if highlight == 1:
        if sqSelected == ():
            drawBoard(screen)
            drawPieces(screen, gs.matrix)

        else:
            r = sqSelected[0]
            c = sqSelected[1]
            p.draw.rect(screen, 'green', p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))
            piece = gs.matrix[r][c]
            screen.blit(IMAGES[piece.get_image_name()], p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))



def drawBoard(screen):

    line_thickness = 2
    p.draw.rect(screen, 'brown', p.Rect(0, 0, WIDTH, HEIGHT))
    for r in range(DIMENSION):
        for c in range(DIMENSION):

            p.draw.rect(screen, 'black', p.Rect(c * SQ_SIZE, r * SQ_SIZE, line_thickness, SQ_SIZE))
            p.draw.rect(screen, 'black', p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, line_thickness))


def drawPieces(screen, board):
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            piece = board[r][c]
            if piece != "empty":
                screen.blit(IMAGES[piece.get_image_name()], p.Rect((c * SQ_SIZE)+SQ_SIZE/2, (r * SQ_SIZE)+SQ_SIZE/2, SQ_SIZE, SQ_SIZE))


main()

