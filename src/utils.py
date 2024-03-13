## LIBRARIES
import ast
import numpy as np
import re


## FUNCTIONS
def read_inputs(dir: str) -> tuple[np.ndarray, dict, dict]:
    ''' 
    Reads input files for Tile Placement puzzle 
    
    Args:
        dir (str): Directory path to input file

    Returns:
        tuple[np.ndarray, dict, dict]: A tuple containing the landscape, tiles, and targets of Tile Placement
    '''

    landscape_flag, tiles_flag, target_flag = False, False, False
    landscape, tiles, targets = [], [], []

    with open(dir, 'r') as input_f:
        for line in input_f:
            # Flags
            if line.startswith("# Landscape"):
                landscape_flag = True
                tiles_flag = False
                target_flag = False
                continue

            elif line.startswith("# Tiles"):
                landscape_flag = False
                tiles_flag = True
                target_flag = False
                continue

            elif line.startswith("# Targets"):
                landscape_flag = False
                tiles_flag = False
                target_flag = True
                continue

            # Append to lists
            if landscape_flag:
                landscape.append(list(line.strip('\n').split(' '))[:-1])    # Gets rid of new line, splits to find blank spaces, and [:-1] to get rid of trailing '' value from input file
            
            elif tiles_flag:
                tiles.append(line.strip('\n'))

            elif target_flag:
                targets.append(line.strip('\n'))

    # Manual truncation
    landscape = landscape[:-1]      # Removes last empty line
    tiles = tiles[0]                # We only care about the tiles metadata
    targets = targets[:4]           # We only care about the four target constraints

    # Process variables into usable format
    landscape, tiles, targets = process_inputs(landscape=landscape, tiles=tiles, targets=targets)

    return landscape, tiles, targets


def process_inputs(landscape: list, tiles: str, targets: list) -> tuple[np.ndarray, dict, dict]:
    """
    Processes inputs into logic usable format

    Args:
        landscape (list): Unprocessed 2D list of landscape
        tiles (str): Unprocessed string of tiles
        targets (list): Unprocessed list of strings of targets
    
    Returns:
        tuple[np.ndarray, dict, dict]: A tuple containing the processed landscape, tiles, and targets
    """
    # Process landscape: Blank spaces in landscape are represented by two values in list ['', ''] b/c of .split(' ') behavior, so merge it into one value
    processed_landscape = []
    
    for item in landscape:
        processing_item = []        # Process items row by row
        skip_next = False

        for char in item:
            if skip_next:
                skip_next = False
                continue
            
            if char == '':
                skip_next = True
                processing_item.append('0')
            else:
                processing_item.append(char)
        
        processed_landscape.append(processing_item)
    
    processed_landscape = np.array(processed_landscape, dtype=int)      # Convert to array

    # Process tiles: String representation of dict to dict
    tiles = tiles.replace('=', ':')
    tiles = re.sub(r'(\b[A-Z_]+\b):', r"'\1':", tiles)
    processed_tiles = ast.literal_eval(tiles)

    # Process targets: List representation of targets to dict
    processed_targets = {int(k): int(v) for k, v in (item.split(':') for item in targets)}
    
    return processed_landscape, processed_tiles, processed_targets


def convert_to_grid_of_grid(array: np.ndarray, block_size: tuple=(4, 4)) -> np.ndarray:
    """
    Converts a 2D array (e.g., landscape) into a grid of grid (overlayed grid over landscape which each block is the area of tiles), where the each grid is a smaller 2D block of given size.
    
    Args:
        array (np.ndarray): Unprocessed 2D array
        block_size (tuple): A tuple for the 2D size of the blocks in the grid of grid.
    
    Returns:
        np.ndarray: A 4D array for the grid of grid.
    """

    # Ensure the array is evenly divisible by block size
    # assert array.shape[0] % block_size[0] == 0, "Row dimension is not divisible by block row size."
    # assert array.shape[1] % block_size[1] == 0, "Column dimension is not divisible by block column size."
    
    # For the bigger grid, calculate the columns and rows
    n_blocks_row = array.shape[0] // block_size[0]
    n_blocks_col = array.shape[1] // block_size[1]
    
    # Convert array into the grid of grid structure, with each grid is being dimensions of block_size
    #
    # .reshape() order:
    #   1) n_block_rows, which is the number of rows in the bigger grid
    #   2) block_size[0], which is the number of rows in each block; each block is a cell in the bigger grid. 
    #   3) n_blocks_col, which is the number of cols in the bigger grid
    #   4) block_size[1], which is the number of cols in each block
    # .swapaxes(1, 2):
    #   Swaps the 2nd and 3rd dimensions into the desired 4D grid structure
    #
    # 4D numpy array format:
    #   1st dim = rows of blocks
    #   2nd dim = cols of blocks
    #   3rd dim = rows within each block
    #   4th dim = cols within each block
    grid_of_grid = (array.reshape(n_blocks_row, block_size[0], n_blocks_col, block_size[1]).swapaxes(1,2))

    return grid_of_grid


def print_grid_of_grid(gog: np.ndarray):
    """
    Prints grid of grid array in a visually intuitive format.

    Args:
        gog (np.ndarray): Grid of grid
    """

    gog_shape = gog.shape

    for i in range(gog_shape[0]):
        for line in range(gog_shape[2]):  # blocks_shape[2] is 4, for each row within a block
            for j in range(gog_shape[1]):
                print(' '.join(map(str, gog[i, j, line])), end=' | ')
            print()
        print('\n' + '-' * 50)


## DRIVER CODE
# dir = "..\\data\\inputs\\tilesproblem_1326658913086500.txt"
# landscape, tiles, targets = read_inputs(dir)
# # print(landscape)
# # print(tiles)
# # print(targets)

# blocks = convert_to_grid_of_grid(landscape)
# # print_grid_of_grid(blocks)
# print(len(blocks))

# rows_of_blocks, cols_of_blocks = blocks.shape[:2]
# domain = {(i, j): ['FULL_BLOCK', 'OUTER_BOUNDARY', 'EL_SHAPE1', 'EL_SHAPE2', 'EL_SHAPE3', 'EL_SHAPE4'] 
#                 for i in range(rows_of_blocks) for j in range(cols_of_blocks)}

# for block in domain:
#     print(block)
# print(domain)

# # Assuming 'blocks' is your 4D numpy array of shape (5, 5, 4, 4)
# blocks_shape = blocks.shape

# # Iterate over each block
# for i in range(blocks_shape[0]):  # Row of blocks
#     for j in range(blocks_shape[1]):  # Column of blocks
#         # Get the current block
#         current_block = blocks[i, j]
        
#         # Get unique values and their counts for the current block
#         unique_values, counts = np.unique(current_block, return_counts=True)
        
#         # Create a dictionary to hold counts for values 1, 2, 3, 4 (excluding 0)
#         value_counts = {value: count for value, count in zip(unique_values, counts) if value != 0}
        
#         # Ensure all expected values are in the dictionary, even if they're not present in the block
#         for value in range(1, 5):
#             if value not in value_counts:
#                 value_counts[value] = 0
                
#         # Print the counts for the current block
#         print(f"Block [{i}, {j}] counts: {value_counts}")

