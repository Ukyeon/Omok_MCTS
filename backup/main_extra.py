from math import floor

import pygame as p

import gameState

WIDTH = 800
HEIGHT = 800
DIMENSION = 15
SQ_SIZE = 600 // DIMENSION
MAX_FPS = 15
IMAGES = {}

# load images 1 time. Reloading images is taxing

'''
Initialize a global dictionary of images. Called 1 time. 
'''


def load_images():
    # access image by IMAGES['wp']
    pieces = ['wp', 'wR', 'wN', 'wB', 'wk', 'wQ', 'bp', 'bR', 'bN', 'bB', 'bK', 'bQ']
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
    playerClicks = []

    while running:

        for e in p.event.get():
            if e.type == p.QUIT:
                running = False

            elif e.type == p.MOUSEBUTTONDOWN:
                location = p.mouse.get_pos()  # tracks mouse xy

                col = (location[0]-100) // SQ_SIZE
                row = (location[1]-100) // SQ_SIZE

                print("column",col)
                print("row", row)
                print("xy", location)

                if sqSelected == (row, col):  # deselect sq
                    sqSelected = ()
                    playerClicks = []

                else:
                    sqSelected = (row, col)
                    playerClicks.append(sqSelected)
                    original = playerClicks[0]
                    from_row = original[0]
                    from_col = original[1]
                    gs.flip_highlight_switch(1)

                    if gs.matrix[from_row][from_col] == 'empty':
                        playerClicks = []
                        gs.flip_highlight_switch(0)

                if len(playerClicks) == 2:
                    original = playerClicks[0]
                    new = playerClicks[1]
                    from_row = original[0]
                    from_col = original[1]

                    to_row = new[0]
                    to_col = new[1]
                    piece = gs.matrix[from_row][from_col]

                    if gs.matrix[from_row][from_col] != "empty":

                        sqSelected = ()
                        playerClicks = []

                        if piece.color == 'white' and player_turn == 0:
                            gs.move(from_row, from_col, to_row, to_col)
                            player_turn = player_turn + 1
                            player_turn = player_turn % 2

                        if piece.color == 'black' and player_turn == 1:
                            gs.move(from_row, from_col, to_row, to_col)
                            player_turn = player_turn + 1
                            player_turn = player_turn % 2


        drawGameState(screen, gs, sqSelected)
        clock.tick(MAX_FPS)
        black = (0, 0, 0)
        myFont = p.font.SysFont("Times New Roman", 35)
        player = ''
        if player_turn == 0:
            p.draw.rect(screen, 'white', p.Rect(300, 30, 200, 35))
            player = 'white'
        else:
            p.draw.rect(screen, 'white', p.Rect(300, 30, 200, 35))
            player = 'black'

        player_display = myFont.render(player, 1, black)
        screen.blit(player_display, (354, 30))
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
            p.draw.rect(screen, 'green', p.Rect((c * SQ_SIZE)+100, (r * SQ_SIZE)+100, SQ_SIZE, SQ_SIZE))
            piece = gs.matrix[r][c]
            screen.blit(IMAGES[piece.get_image_name()], p.Rect((c * SQ_SIZE)+100, (r * SQ_SIZE)+100, SQ_SIZE, SQ_SIZE))


def drawBoard(screen):
    colors = [p.Color("white"), p.Color("gray")]
    #p.draw.rect(screen, 'black', p.Rect(90, 557, SQ_SIZE, SQ_SIZE))
    p.draw.rect(screen, 'black', p.Rect(90, 90, 620, 620))
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            color = 'brown' # colors[((r + c) % 2)]
            p.draw.rect(screen, color, p.Rect((c * SQ_SIZE)+100, (r * SQ_SIZE)+100, SQ_SIZE, SQ_SIZE))



def drawPieces(screen, board):
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            piece = board[r][c]
            if piece != "empty":
                screen.blit(IMAGES[piece.get_image_name()], p.Rect((c * SQ_SIZE)+100, (r * SQ_SIZE)+100, SQ_SIZE, SQ_SIZE))


main()
# print('%d, %d' % (col, row))

'''
def get_rc(location):

    x = location[0]
    y = location[1]
    rc = [-1 , -1]

    if 100 <= y < 175:
        rc[0] = 0
    if 175 <= y < 250:
        rc[0] = 1
    if 250 <= y < 325:
        rc[0] = 2
    if 325 <= y < 400:
        rc[0] = 3
    if 400 <= y < 475:
        rc[0] = 4
    if 475 <= y < 550:
        rc[0] = 5
    if 550 <= y < 625:
        rc[0] = 6
    if 625 <= y < 700:
        rc[0] = 7

    if 100 <= x < 175:
        rc[1] = 0
    if 175 <= x < 250:
        rc[1] = 1
    if 250 <= x < 325:
        rc[1] = 2
    if 325 <= x < 400:
        rc[1] = 3
    if 400 <= x < 475:
        rc[1] = 4
    if 475 <= x < 550:
        rc[1] = 5
    if 550 <= x < 625:
        rc[1] = 6
    if 625 <= x < 700:
        rc[1] = 7

    return rc
'''