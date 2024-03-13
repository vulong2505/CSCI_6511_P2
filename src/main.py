from tile_placement import TilePlacement
from csp_solver import *
from multiprocessing import Process, Manager, cpu_count
from time import time, sleep
import copy
import os


## Functions
def get_data_path(data_folder: str='inputs', filename: str=None) -> str:
    """ Relative path to data files """
    current_file_path = os.path.abspath(__file__)
    top_level_directory = os.path.dirname(os.path.dirname(current_file_path))

    return os.path.join(top_level_directory, 'data', data_folder, filename)


def run_solver_process(csp_solver: CSP, return_dict: dict, key: int):
    """ Run solver and store solution in referenced return dictionary """

    try:
        time_start = time()
        solution = csp_solver.begin_backtracking()
        duration = round(time() - time_start, 4)

        if solution and csp_solver.verify(solution):
            return_dict[key] = (solution, duration)

    except Exception as e:
        # Handle exceptions if necessary
        print(f"Process encountered an error: {e}")


def find_solutions_in_parallel(csp_solver: CSP, timeout: int=300, num_processes: int=4) -> tuple[dict, int]:
    """ Run CSP problem in parallel, and return first solution path found. """

    # Init Manager
    manager = Manager()
    return_dict = manager.dict()

    # Start processes
    processes = []
    for i in range(num_processes):
        p = Process(target=run_solver_process, args=(copy.deepcopy(csp_solver), return_dict, i))
        p.start()
        processes.append(p)
    
    # Keep track of timeout
    start_time = time()
    while time() - start_time < timeout:
        if return_dict:
            # A solution is found, so break the loop and proceed to terminate all processes
            break
        sleep(0.1)  # Sleep briefly to avoid hogging the CPU

    # Forcefully terminate all processes
    for p in processes:
        p.terminate()  

    if return_dict:
        # Return the first solution found
        for key in return_dict.keys():
            return return_dict[key]
    
    print("No solution found within the timeout. Try again?")
    return None


## MAIN
if __name__ == '__main__':

    # Get dir    
    data_folder = input("Enter data folder: ")
    fname = input("Enter file name: ")

    if data_folder != "generated":
        dir = get_data_path(filename=fname)
    else:
        dir = get_data_path(data_folder=data_folder, filename=fname)


    try:
        problem = TilePlacement(dir)
        csp_solver = CSP(problem, DEBUG_PRINT=False)
        # # Run one instance of CSP solver
        # print("Starting CSP solver...")
        # time_start = time()
        # soln_assignment = csp_solver.begin_backtracking()
        # time_end = time()
        # print("Time used: ", round(time_end - time_start, 4))
        # print("Soln:", soln_assignment)
        # print(csp_solver.verify(soln_assignment)) # Bool to check if solution is correct or not

        # Multithread to run multiple instances of CSP solver, returns the first solution found (earliest time).
        cores = None
        while cores is None:
            cores = input("Enter # of threads (1 thread is the same as running non-parallel): ")
            try:
                cores = int(cores)
            except ValueError:
                print("Try entering a valid integer.")
                cores = None

        print("Starting CSP solver threads...")
        solution = find_solutions_in_parallel(csp_solver, timeout=300, num_processes=cores)
        if solution is not None:
            print("Time used:", solution[1])
            print("Solution:", solution[0])


    except Exception as e:
        print(e)
        print("Unable to find input file... Try again.")


