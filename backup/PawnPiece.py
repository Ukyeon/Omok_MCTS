class PawnPiece:

    def __init__(self, row, col, color):
        self.row = row
        self.col = col
        self.color = color

    def get_image_name(self):

        if self.color == "black":
            return 'bC'

        else:
            return 'wC'

    def move(self, row, col):
        self.row = row
        self.col = col
