# IDeLM-Glulam

This repository contains the code for the glulam optimizer developed as part of the IDeLM project.
The optimizer is based on the [Gurobi](https://www.gurobi.com/) optimization solver and is written in Python.
The optimizer is designed to optimize the cutting pattern and packaging of glulam beams.

## Installation

To set up the project on your local machine:

1. Clone the repository:
    ```
    git clone git@github.com:HI-IDN/IDeLM-Glulam.git
    ```
2. Navigate to the project directory:
    ```
    cd IDeLM-Glulam
    ```
3. Install the dependencies:
    ```
    pip install -r requirements.txt
    ```

## Usage

Run the optimizer:

```
python3 main.py --depth 115 --file data/glulam.csv  
```

## Configuration

If your project involves configuration settings then they should be altered in the `config/settings.py` file.
Default settings are provided in the file.
Example data file `glulam.csv` is provided in the `data/` directory. It contains the dimensions of the glulam beams and
the number of beams required for each order. There are 5 orders in the example data file. All measurements are in
millimeters. Each press must have the same depth. From the example data file, the depth can be 90mm, 115mm, 140mm, 160mm
or 185mm. The depth is specified as a command line argument when running the `main.py`. The default depth is 115mm.

## Project Structure
```
IDeLM-Glulam/
│
├── config/
│   └── settings.py             # Configuration file for glulam dimensions and settings
│
├── data/
│   └── glulam.csv              # Default data file
│
├── models/
│   ├── cutting_pattern.py      # Code for cutting pattern generation (Gurobi MIP)
│   └── pack_n_press.py         # Code for packaging optimization (Gurobi MIP)
│
├── strategies/
│   └── evolution_strategy.py   # Code for evolution strategy (1+1 ES)
│
├── utils/
│   ├── data_processor.py       # Utility for processing data file
│   └── test_suite.py           # Code for unit tests for the code
│
├── .gitignore                  # File specifying untracked files to ignore
├── README.md                   # Project description and documentation
├── requirements.txt            # List of dependencies for the project
└── main.py                     # Main script to run the optimizer
```