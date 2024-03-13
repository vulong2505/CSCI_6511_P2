import numpy as np
from utils import read_inputs, convert_to_grid_of_grid

class TilePlacement(): 
    """ A class for Time Placement logic. """
    
    def __init__(self, input_file: str):
        """
        Initializes Time Placement problem with the given input file.

        Parameters:
            input_file (str): Directory path to input file with problem metadata
        """

        # Given input file
        self.input_file = input_file

        # Produce landscape, tiles constraint, visibility constraint
        self.LANDSCAPE, self.TILES, self.TARGETS = self.load_data()

        # Convert to grid of grid
        self.BLOCKS = convert_to_grid_of_grid(self.LANDSCAPE)
        
        # CSP data
        self.BLOCKS_VARS_LEN = len(self.BLOCKS) * len(self.BLOCKS)  # this will be the number of blocks in the grid of grid


    def load_data(self):
        """
        Loads the input file and returns Time Placement metadata.

        Returns:
            tuple[np.ndarray, dict, dict]: A tuple containing the landscape, tiles, and targets metadata
        """

        try:
            landscape, tiles, targets = read_inputs(self.input_file)
        except FileNotFoundError:
            print("Error: File not found.")
        except Exception as e:
            print(f"Error: {e}")

        return landscape, tiles, targets
    
    
    def get_bush_count_in_block(self, block: np.ndarray) -> int:
        """
        Counts instances of each bush types in specific block[i][j] 
        
        Args:
            block (np.ndarray): Block at row i, column j in the grid of grid. 
        
        Returns:
            (dict): Dictionary with (bush_type: count) as its key-value pair.
        """

        unique_values, counts = np.unique(block, return_counts=True)
        
        # Create a dictionary to hold counts for values 1, 2, 3, 4 (excluding 0)
        value_counts = {bush_color: count for bush_color, count in zip(unique_values, counts) if bush_color != 0}
        
        # Ensure all expected values are in the dictionary, even if they're not present in the block
        for value in range(1, 5):
            if value not in value_counts:
                value_counts[value] = 0

        return value_counts


    def get_total_bush_count_in_block(self, block: np.ndarray) -> int:
        """
        Counts all the bushes in the block

        Args:
            block (np.ndarray): Block at row i, column j in the grid of grid. 
        
        Returns:
            (int): Total bushes in the block
        """
    
        unique_values, counts = np.unique(block, return_counts=True)

        # Sum up all counts of bushes
        count_sum = 0
        
        for bush_type, count in zip(unique_values, counts):
            # Skips 0s explicitly
            if bush_type == 0:
                continue
            else:
                count_sum += count

        return count_sum