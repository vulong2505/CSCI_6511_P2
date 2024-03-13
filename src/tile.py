class Tile:
    """
    A class for tile shape logic

    """

    def __init__(self):
        self.TILE_SIZE = 4
        self.tile_shape = None

    def getTileShape(self):
        return self.tile_shape
    
    def setTileShape(self, tile_shape_arg):
        self.tile_shape = tile_shape_arg

    def cover(self, i: int, j: int) -> bool:
        pass