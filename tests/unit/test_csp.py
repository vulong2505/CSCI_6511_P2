import unittest
import os
from tile_placement import TilePlacement
from csp_solver import *


def get_data_path(filename):
    """ Relative path to data/inputs files """

    current_file_path = os.path.abspath(__file__)
    top_level_directory = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    return os.path.join(top_level_directory, 'data', 'inputs', filename)


class TestCSPSolver(unittest.TestCase):
    """ Unit test class to test input files given by professor """

    def test_input1(self):
        # Initialize
        dir = get_data_path('input1.txt')
        problem = TilePlacement(dir)
        csp_solver = CSP(problem)

        # Run
        soln_assignment = csp_solver.backtrack()

        # Verify
        self.assertEqual(csp_solver.verify(soln_assignment), True)


    def test_input2(self):
        # Initialize
        dir = get_data_path('input2.txt')
        problem = TilePlacement(dir)
        csp_solver = CSP(problem)

        # Run
        soln_assignment = csp_solver.backtrack()

        # Verify
        self.assertEqual(csp_solver.verify(soln_assignment), True)


    def test_input3(self):
        # Initialize
        dir = get_data_path('input3.txt')
        problem = TilePlacement(dir)
        csp_solver = CSP(problem)

        # Run
        soln_assignment = csp_solver.backtrack()

        # Verify
        self.assertEqual(csp_solver.verify(soln_assignment), True)


if __name__ == "__main__":
    unittest.main()