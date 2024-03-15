# Tile Placement
Solving Tile Placement with CSP

# Instructions to Run
1) Run src/main.py
2) The program will first ask for the location of the input text file for the puzzle. You may use the three input files (input1.txt, input2.txt, input3.txt) or create your own. You may find the data input files in data/inputs or if you may create your own data/generated folder.
- In the command line, it will ask ```Enter data folder:```, referring to either the data/inputs folder or the data/generated folder if you have generated files. You may enter either "inputs" or "generated". 
- Then, the command line will ask ```Enter file name:```, referring to the input text file name, e.g. input1.txt.
3) After providing the input text file, the program will ask ```Enter # of threads (1 thread is the same as running non-parallel):```. Default is 1 for running one instance of the CSP solver. If your computer is able, running multiple threads equal to the cores you have may speed up performance by increasing the chance the CSP solver finds a valid solution.

# Problem Description
**Given**
* You are given a landscape on which certain “bushes” grow, marked by colors: 1, 2, 3, 4.
* The landscape is of square shape, so, it might be 100 x 100 or 200 x 200 etc.
* You are given a set of “tiles” which are of three different shapes. The tiles are 4 x 4.  One tile only covers part of the landscape.  Here are the shapes
- Full Block: A “full block” tile covers the full 4 x 4 area, and no bush is then visible in that patch.
- An “outer boundary” tile covers the outer boundary of the 4 x 4 area, and any bush in the middle part is visible.
- An “EL” shaped tile covers only two sides of the area.
* You are given a “target” of which bushes should be visible after you have finished placing the tiles.

**Observations**
* The total tiles cover the entire landscape.  However, depending on which tiles are placed where, different parts of the landscape, and hence different bushes are visible.
* The number of tiles equals the size of the area divided by the size of tile.  So, for 20 x 20 landscape, you are given 25 tiles.

**Input Files**
* Structure of the input file is as follows.
- Landscape is given in a space delimited, new line separated.
- Tiles in terms of counts by different shapes.
- Target of how many different bushes should be visible.
