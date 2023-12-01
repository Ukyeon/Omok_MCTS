from PawnPiece import PawnPiece

w, h = 15, 15
MATRIX = [[0 for x in range(w)] for y in range(h)]

for i in range(h):
    for j in range(h):
        MATRIX[i][j] = "empty"
bC = PawnPiece(1, 0, 'black')
wC = PawnPiece(1, 0, 'white')
MATRIX[0][0] = bC
MATRIX[2][3] = wC

class GameState:
    def __init__(self):
        self.matrix = MATRIX
        self.highlightSwitch = 0

    def move(self, from_row, from_col, to_row, to_col):

        if self.matrix[from_row][from_col] != "empty":
            piece = self.matrix[from_row][from_col]
            self.matrix[from_row][from_col] = "empty"
            self.matrix[to_row][to_col] = piece

    def flip_highlight_switch(self, value):
        self.highlightSwitch = value

    def get_highlight_switch(self):
        return self.highlightSwitch

def test():

    gs = GameState()

    x = 0
    print(x)
    test_reference(x)
    print(x)


def test_reference(x):

    return x+1




test()