 # RNA Comparison
 
 ## Description
 Compares various RNA sequences using the Zhang–Sasha tree edit distance algorithm. 
 
 ## Dependencies
 - numpy
 - hmmlearn
 - zss
 - tqdm
 - graphviz
 - scikit-bio (skbio)
 - pandas
 
 ## Instaling Dependencies
 
 ```bash
 pip install numpy hmmlearn zss tqdm graphviz scikit-bio pandas
 ```
 
 
 ## Running
 Navigate to the `./Code` directory:
 
 ```bash
 cd Code
 ```
 
 Run the main results:
 
 ```bash
 python main.py
 ```
 
 Run the analysis:
 
 ```bash
 python analysis.py
 ```
 
 - Results of the **PERMANOVA test** will be displayed on screen after analysis.
 - Case-by-case results will be saved to `./Code/results.csv`.
 
 ## Bulk testing
 To add your own files for bulk testing:
 1. Navigate to the `./CMs` directory.
 2. Create a new subdirectory.
 3. Fill it with `.cm` files.
 
 All `.cm` files in the same subdirectory will be assumed to belong to the same RNA family.
 
 ## Directory structure
 ```
 rna-comparison/
 ├── Code/                          # Core scripts
 │   ├── main.py                    # Entry point for RNA comparison
 │   ├── analysis.py                # Statistical analysis (PERMANOVA)
 │   ├── Structures.py              # RNA structure definitions
 │   ├── SubStructures.py           # RNA substructure logic
 │   ├── TraversalMechanisms.py     # Tree traversal algorithms
 │   ├── ComplexParser.py           # Main parser for CM's
 │   └── +Supplementaryfiles        # Support functions
 ├── CMs/                           # RNA family directories (.cm files)
 └── README.md

