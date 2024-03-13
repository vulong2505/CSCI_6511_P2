from tile_placement import TilePlacement
from copy import deepcopy
from collections import Counter
from sortedcontainers  import SortedSet
from operator import neg
import numpy as np
import random


class CSP:
    """ Constraint Satisfactory Problem Solver """


    def __init__(self, tile: TilePlacement, DEBUG_PRINT: bool=False):
        """
        Initializes CSP with Tile Placement problem

        Args:
            tile (TilePlacement): Tile Placement problem instance
            DEBUG_PRINT (bool): Used for debugging
        """


        # Delete later
        self.DEBUG_PRINT = DEBUG_PRINT

        # Tile Placement problem
        self.TILE_PROBLEM = tile

        # Variables which represent each block in the grid of grid
        self.variables = tile.BLOCKS
        self.BLOCK_VARS_LEN = tile.BLOCKS_VARS_LEN

        # Domains of each block
        rows_of_blocks, cols_of_blocks = tile.BLOCKS.shape[:2]
        self.domain = {(i, j): ['OUTER_BOUNDARY', 'EL_SHAPE1', 'EL_SHAPE2', 'EL_SHAPE3', 'EL_SHAPE4', 'FULL_BLOCK'] 
                       for i in range(rows_of_blocks) for j in range(cols_of_blocks)}
        
        # Neighbors for AC3
        self.neighbors = {(i, j): ['OUTER_BOUNDARY', 'EL_SHAPE1', 'EL_SHAPE2', 'EL_SHAPE3', 'EL_SHAPE4', 'FULL_BLOCK'] 
                            for i in range(rows_of_blocks) for j in range(cols_of_blocks)}
        
        # Constraint 1: Number of each tile type is available
        self.tile_constraints = tile.TILES

        # Constraint 2: Visibility requirements for each bush type
        self.visibility_constraints = tile.TARGETS


        # Puzzle logic
        self.visible_bushes = {1: 0, 2: 0, 3: 0, 4: 0}
        self.init_visible_bushes(tile.BLOCKS) 

    # =============================================================================
    # Backtracking

    # def backtracking_search(self, select_unassigned_variable=None, order_domain_values=None, inference=None):
    #     """ Configure backtracking search. """
        
    #     # Default variable select is MRV
    #     if select_unassigned_variable is None:
    #         select_unassigned_variable = self.MRV

    #     # Default val select is LCV
    #     if order_domain_values is None:
    #         order_domain_values = self.LCV

    #     # Default inference is MAC
    #     if inference is None:
    #         inference = MAC

    #       ... backtrack subfunction here

    #     return self.backtrack({})


    def backtrack(self, assignment=None) -> dict:
        """
        (Currently Naive) Backtracking algorithm for CSP

        Args:
            assignment (dict): Assignment of values to variables

        Returns:
            (dict): Complete assignment of values to variables; solution to CSP problem.
        """

        if assignment is None:
            assignment = {}

        # Goal test: Check if every variable has been assigned a value. Base case to recursion.
        if len(assignment) == self.BLOCK_VARS_LEN:
            return assignment

        # Delete later
        if self.DEBUG_PRINT:
            print('Current assignment:', assignment)

        # Ordering by MRV
        var = self.MRV(assignment)
        
        # Ordering by LCV
        for value in self.LCV(var, assignment):
            # Successor function: Check if current tile value breaks constraint
            if self.is_consistent(var, value, assignment):
                
                # Keep track of assignments
                assignment[var] = value
                
                inference_candidates = self.get_inference_candidates(var, value)

                # Recurse                
                if self.MAC(var, assignment, inference_candidates):
                    result = self.backtrack(assignment)

                    # Propogate successful solution steps back up recursion call stack 
                    if result is not None:
                        return result

                # Restore domain
                self.restore_curr_domain(inference_candidates)
                                    
                if self.DEBUG_PRINT:
                    print('Backtracking...', var, assignment[var])
                
                # Backtracking: Undo the move if the current result leads to nothing
                del assignment[var]

        return None


    def naive_select_var(self, assignment: dict) -> tuple[int, int]:
        """ 
        Idea 1: Pick one block at a time.
        
        TODO: Implement MRV  

        Args:
            assignment (dict): In progress assignment of variables

        Returns:
            tuple[int, int]: The (i, j) location of a block not yet in assignment
        """

        # Remember, dict has key-pair items {(i, j): values}. Iterating through block in self.domain returns (i, j) of each block in domain.
        for block in self.domain:
            if block not in assignment:
                return block
            
        # Assignment finished
        return None 


    def naive_select_val(self, var: tuple[int, int], assignment: dict) -> list:
        """ 
        Return the domain values of the var. Doesn't order it smartly, just returns the domain values as is. Naive method.
        
        Args:
            var (tuple[int, int]): The (i, j) location of a block
            assignment (dict): Current assignment of values to variables

        Returns:
            (list): List of values in the domain. For Tile Placement, list is in form ['FULL_BLOCK', 'OUTER_BOUNDARY', 'EL_SHAPE1', 'EL_SHAPE2', 'EL_SHAPE3', 'EL_SHAPE4'].
        """

        return self.domain[var]


    def is_consistent(self, var: tuple[int, int], value: str, assignment: dict) -> bool:
        """ 
        Returns a bool if the assignment is consistent
        
        Args:
            var (tuple[int, int]): Index of variable
            value (str): Tile shape
            assignment: Current assignment

        Returns
            (bool): Is consistent or not
        """

        n_of_conflicts = self.n_of_conflicts(var, value, assignment)

        # if self.DEBUG_PRINT:
        #     print('Conflicts for (var, value):', var, value, '->', n_of_conflicts, 'conflicts')

        if n_of_conflicts == 0:
            return True
        else:
            return False


    def n_of_conflicts(self, var: tuple[int, int], value: str, assignment: dict) -> int:
        """
        Count the amount of constraints violated.
        
        Args:
            var (tuple[int, int]): Index of variable
            value (str): Tile shape
            assignment (dict): Current assignment

        Returns:
            (int): Count of conflicts
        """

        # Initialize count of constraint conflicts
        conflict_count = 0

        # Temp add var into value
        temp_assignment = assignment.copy()
        temp_assignment[var] = value

        # Check Constraint 1: Finite number of tiles
        # Assignment has all the tiles placed thus far. I can count all the tiles in assignment, and compare them to each of the finite tile constraint
        check_constraint_1 = {}

        for tile_value in temp_assignment.values():
            if ('EL_SHAPE' in tile_value):      # Sum 'EL_SHAPE1', 'EL_SHAPE2', 'EL_SHAPE3', 'EL_SHAPE4' as a total count
                check_constraint_1['EL_SHAPE'] = check_constraint_1.get('EL_SHAPE', 0) + 1
            else:
                check_constraint_1[tile_value] = check_constraint_1.get(tile_value, 0) + 1

        for tile_value, count in check_constraint_1.items():
            if (count > self.tile_constraints[tile_value]):
                conflict_count += 1

        
        # Check Constraint 2: Target of "exact" number of bushes needs to be visible after all tiles has placed.
        check_constraint_2 = deepcopy(self.visible_bushes)
        
        for block_idx, tile_value in temp_assignment.items():
            # i, j = block_idx

            # Decrement bushes covered by tile_value in this block
            self.update_visible_bushes(block_idx, tile_value, check_constraint_2)

        # if check_constraint_2 is less than targets for each bush, increment conflict count
        for bush_color, target in self.visibility_constraints.items():
            
            # If the assignment is full, check if counts are exactly equal to visiblity_constraint targets
            if len(temp_assignment) == self.BLOCK_VARS_LEN:
                if check_constraint_2[bush_color] != target:
                    conflict_count += 1
            
            # Otherwise, check if current value assignment to variable break constraints
            if check_constraint_2[bush_color] < target:
                conflict_count += 1
            
            
        # Return the count: 0 is no conflicts, >0 means constraints are broken
        return conflict_count
        
    # =============================================================================
    # Heuristic Methods
    
    def MRV(self, assignment: dict) -> tuple[int, int]:
        # Ordering: which variable (e.g., the states, the block in the grid of grid) to pick
        # Minimum remaining variable/Most constrained variable
        # Fail-fast ordering
        ''' 
        Select next unassigned block with the minimum count of bushes; tie-break by through random pick. 
        Fail-fast ordering: By picking minimum count, the ordering will have blocks with higher counts later on in the search, leaving the solver with less/minimal remaining values.

        Args:
            assignment (dict): Current assignment

        Returns:
            (tuple[int, int]): Index of variable picked by MRV
        '''

        # Lambda function to count bushes
        f = lambda var: self.count_of_bushes(var)

        # Return list of (i, j) coordinates of unassigned blocks
        unassigned_blocks = [v for v in self.domain if v not in assignment]

        # Return list of bush count for each (i, j) coordinate
        unassigned_blocks_count = [f(self.variables[i][j]) for i, j in unassigned_blocks]

        # Get max value of bush count in unassigned blocks
        min_bush_count = min(unassigned_blocks_count)

        # Identify blocks that have this maximum bush count
        min_bush_blocks = [unassigned_blocks[i] for i, count in enumerate(unassigned_blocks_count) if count == min_bush_count]

        # Randomly select one of these blocks if there's a tie
        if min_bush_blocks:
            chosen_block = random.choice(min_bush_blocks)
            return chosen_block
        else:
            return None


    def count_of_bushes(self, var: np.ndarray) -> int:
        '''
        Return one count of bushes in the block, excluding 0.

        Args:
            var (np.ndarray): A block

        Returns:
            (int): Count of bushes in the block
        '''

        return np.sum(var > 0)


    def LCV(self, var: tuple[int, int], assignment: dict) -> list:
        """ 
        Sorts values by ascending order of conflict counts for each variable. 
        Fail-last ordering.
        
        Args:
            var (tuple[int, int]): Index of variable
            assignment (dict): Current assignment
        
        Returns
            (list): Ordered values by LCV
        """        

        # Lambda function to calculate conflict counts
        f = lambda val: self.n_of_conflicts(var, val, assignment)

        # Tuple with tile shape and associated conflict count
        value_conflict_pairs = [(v, f(v)) for v in self.domain[var]]
        
        # Sort the list of tuples based on the conflict counts (the second element of each tuple)
        sorted_value_conflict_pairs = sorted(value_conflict_pairs, key=lambda pair: pair[1])
        
        # Extract and return the sorted list of values
        # sorted_values = [pair[0] for pair in sorted_value_conflict_pairs]

        ## Optimization: Remove any values with conflicts. Technically, we could remove the check for is_consistent() in the backtracking algorithm because of this as well.
        sorted_values_only_zero_conflict_pairs = [pair for pair in sorted_value_conflict_pairs if pair[1] == 0]
        sorted_values_extract_tile_shape = [pair[0] for pair in sorted_values_only_zero_conflict_pairs]

        # if self.DEBUG_PRINT:
        #     print('LCV:', sorted_value_conflict_pairs)
        #     print('LCV Processed:', sorted_values_only_zero_conflict_pairs)

        # return sorted_values
        return sorted_values_extract_tile_shape


    # =============================================================================
    # Constraint Propagation

    def satisfy_constraint(self, assignment: (dict)) -> bool:
        """ 
        Check if the assignment is consistent; not assigning val to var. 
        
        Args:
            assignment (dict): Current assignment
        
        Returns
            (bool): Boolean if assignment satisfies constraints
        """

        # Initialize count of constraint conflicts
        conflict_count = 0

        # Temp add var into value
        temp_assignment = assignment.copy()

        # Check Constraint 1: Finite number of tiles
        # Assignment has all the tiles placed thus far. I can count all the tiles in assignment, and compare them to each of the finite tile constraint
        check_constraint_1 = {}

        for tile_value in temp_assignment.values():
            if ('EL_SHAPE' in tile_value):      # Sum 'EL_SHAPE1', 'EL_SHAPE2', 'EL_SHAPE3', 'EL_SHAPE4' as a total count
                check_constraint_1['EL_SHAPE'] = check_constraint_1.get('EL_SHAPE', 0) + 1
            else:
                check_constraint_1[tile_value] = check_constraint_1.get(tile_value, 0) + 1

        for tile_value, count in check_constraint_1.items():
            if (count > self.tile_constraints[tile_value]):
                conflict_count += 1

        
        # Check Constraint 2: Target of "exact" number of bushes needs to be visible after all tiles has placed.
        check_constraint_2 = deepcopy(self.visible_bushes)
        
        for block_idx, tile_value in temp_assignment.items():
            # i, j = block_idx

            # Decrement bushes covered by tile_value in this block
            self.update_visible_bushes(block_idx, tile_value, check_constraint_2)

        # if check_constraint_2 is less than targets for each bush, increment conflict count
        for bush_color, target in self.visibility_constraints.items():
            
            # If the assignment is full, check if counts are exactly equal to visiblity_constraint targets
            if len(temp_assignment) == self.BLOCK_VARS_LEN:
                if check_constraint_2[bush_color] != target:
                    conflict_count += 1
            
            # Otherwise, check if current value assignment to variable break constraints
            if check_constraint_2[bush_color] < target:
                conflict_count += 1
            
            
        # Return boolean: 0 is no conflicts, >0 means constraints are broken
        if conflict_count == 0:
            return True
        else:
            return False
       

    def restore_curr_domain(self, inference_candidates: list):
        """ Restore inference candidates into current domain """

        for var, value in inference_candidates:
            self.domain[var].append(value)


    def get_inference_candidates(self, var: tuple[int, int], value: str) -> list:
        """ Return inference candidates for assigning value to var """

        # Domain for current variable excluding current value
        inference_candidates = [(var, v) for v in self.domain[var] if v != value]
        
        # Assign value to domain
        self.domain[var] = [value]

        return inference_candidates


    def prune(self, var: tuple[int, int], value: str, inference_candidates: list):
        """ Prune value from domain of var. """

        self.domains[var].remove(value)

        if inference_candidates is not None:
            inference_candidates.append((var, value))


    def map_X(self, Xi):
        """ neighbors """
        if Xi.startswith('OUTER_BOUNDARY'):
            return (0, 0)
        elif Xi.startswith('EL_SHAPE'):
            return (0, 1)
        else:
            return (0, 2)


    def MAC(self, var: tuple[int, int], assignment: dict, inference_candidates: list) -> bool:
        """ 
        Maintain Arc Consistency: runs inside backtracking search. 
        
        Args:
            var (tuple[int, int]): Index of variable
            assignment (dict): Current assignment
            inference_candidates (list): Candidates to check
        
        Returns
            (bool): Inference possible
        """

        # Subfunction for AC3
        def AC3(assignment, queue=None, inference_candidates=None):
            """ Arc Consistency Algorithm """

            # Sort queue
            queue = SortedSet(queue, key=lambda t: neg(len(self.domain[t[1]])))      

            # Keeps track of queue/agenda of constraints to be processed
            while queue:
                (Xi, Xj) = queue.pop() 
                revised = self.revise(Xi, Xj, inference_candidates, assignment)

                if revised:
                    if len(self.domain[Xi]) == 0:
                        return False
                    
                    for Xk in self.neighbors[Xi]:
                        if Xk != Xj:
                            queue.add((Xk, Xi))

            return True
        
        # Run MAC
        result = AC3(assignment, {(X, var) for X in self.neighbors[var]}, inference_candidates)

        return True
    

    def revise(self, Xi, Xj, inference_candidates: list, assignment: dict) -> bool:
        """ Revise is true if X is removed, therefore re-add neighbors of X to queue """

        Xi = self.map_X(Xi)

        revised = False

        for x in self.domain[Xi]:
            # An arc X -> Y is consistent iff for every x in the tail there is some y in the head which could be assigned without violating a constraint
            # Pseudocode: 
            #   if no value y in Dj allows (x,y) to satisfy the constraint between Xi and Xj then
            #       delete x from Di
            #       revised = True

            conflict = True

            for y in self.domain[Xj]:
                if self.satisfy_constraint(assignment):
                    conflict = False
                
                if not conflict:
                    break
            
            if conflict:
                self.prune(Xi, x, inference_candidates)
                revised = True
        
        return revised
            

    # =============================================================================
    # Verify solution

    def verify(self, soln_assignment: dict) -> bool:
        '''
        Verify the solution. 
        
        Args:
            soln_assignment (dict): Solution to Tile Placement puzzle

        Returns:
            (bool): Bool determining if solution truly satisfies constraints
        '''
    
        ## Check if Constraint 1 satisfied
        # Preprocess values to group EL_SHAPE variations
        preprocessed_values = ['EL_SHAPE' if value.startswith('EL_SHAPE') else value for value in soln_assignment.values()]

        # Get unique counts for each tile values
        value_counts = Counter(preprocessed_values)

        for tile_value, count in self.tile_constraints.items():
            if value_counts[tile_value] > count:
                return False
            

        ## Check if Constraint 2 satisfied: visible bushes exactly equal to targets
        visible_bushes = deepcopy(self.visible_bushes)

        for block_idx, tile_value in soln_assignment.items():
            # i, j = block_idx

            # Decrement bushes covered by tile_value in this block
            self.update_visible_bushes(block_idx, tile_value, visible_bushes)


        for bush_color, count in visible_bushes.items():
            if self.visibility_constraints[bush_color] != count:
                return False

        # Solution is correct
        return True


    # =============================================================================
    # Puzzle logic

    # TilePlacement logic
    def update_visible_bushes(self, var: tuple[int, int], value: str, check_constraints_2: dict):
        '''
        Decrement bushes covered by the value in a block from self.visible_bushes 
        
        Args:
            var (tuple[int, int]): Index of variable
            value (str): Tile shape
            check_constraints_2 (dict): Visibility of bushes
        '''

        covered_bushes = self.determine_covered_bushes(var, value)

        # print('check_constraint_2', check_constraints_2)
        # print('covered:', covered_bushes)

        for bush_color, count in covered_bushes.items():
            check_constraints_2[bush_color] -= count
    

    def determine_covered_bushes(self, var: tuple[int, int], value: str) -> dict:
        '''
        Return a dict of the covered bushes given a value/tile shape of a variable/block.
        
        Args:
            var (tuple[int, int]): Index of variable
            value (str): Tile shape

        Returns:
            (dict): Covered bushes based on assignment of val to var
        '''

        # Get the block pertaining to the var
        var = self.variables[var]

        # if statement for val = [FULL_BLOCK, OUTER_BOUNDARY, EL_SHAPE1, EL_SHAPE2, EL_SHAPE3, EL_SHAPE4]
        covered_bushes = {1: 0, 2: 0, 3: 0, 4: 0}
        
        # Define the mask for each tile shape
        if value == "FULL_BLOCK":
            mask = np.ones((4, 4), dtype=bool)  # All covered
        elif value == "OUTER_BOUNDARY":
            mask = np.ones((4, 4), dtype=bool)  # All visible initially
            mask[1:-1, 1:-1] = False  # Middle 2x2 is not covered
        elif value == "EL_SHAPE1":
            mask = np.zeros((4, 4), dtype=bool)  # All visible initially
            mask[:, 0] = True  # Left column covered
            mask[-1, :] = True  # Bottom row covered
        elif value == "EL_SHAPE2":
            mask = np.zeros((4, 4), dtype=bool)  # All visible initially
            mask[:, -1] = True  # Right column covered
            mask[-1, :] = True  # Bottom row covered
        elif value == "EL_SHAPE3":
            mask = np.zeros((4, 4), dtype=bool)  # All visible initially
            mask[0, :] = True  # Top row covered
            mask[1:, -1] = True  # Right column except top cell covered
        elif value == "EL_SHAPE4":
            mask = np.zeros((4, 4), dtype=bool)  # All visible initially
            mask[0, :] = True  # Top row covered
            mask[1:, 0] = True  # Left column except top cell covered
        

        # Now, calculate covered bushes    
        for bush_color in range(1, 5):
            # Count only covered bushes of each type
            covered_bushes[bush_color] = np.sum(var[mask] == bush_color)

        return covered_bushes


    def init_visible_bushes(self, blocks: np.ndarray):
        """ Initializes count of all visible bushes. """
        for i in range(len(blocks)):
            for j in range(len(blocks[0])):
                # Get count of each bush types in block
                value_counts = self.get_bush_count_in_block(blocks[i][j])

                # Add to self.visible_bushes for initialization
                for key, value in value_counts.items():
                    self.visible_bushes[key] += value


    def get_bush_count_in_block(self, block: np.ndarray) -> dict:
        """
        Counts instances of each bush types in specific block[i][j] 
        
        Args:
            block (np.ndarray): Block at row i, column j in the grid of grid. 
        
        Returns:
            (dict): Dictionary with (bush_color: count) as its key-value pair.
        """

        unique_values, counts = np.unique(block, return_counts=True)
        
        # Create a dictionary to hold counts for values 1, 2, 3, 4 (excluding 0)
        value_counts = {bush_color: count for bush_color, count in zip(unique_values, counts) if bush_color != 0}
        
        # Ensure all expected values are in the dictionary, even if they're not present in the block
        for value in range(1, 5):
            if value not in value_counts:
                value_counts[value] = 0

        return value_counts
