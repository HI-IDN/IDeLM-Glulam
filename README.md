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

## Gurobi Licensing

To use Gurobi, you'll need to log in to their platform and set up a license on your machine. Detailed instructions for
obtaining and setting up the license can be
found [here](https://support.gurobi.com/hc/en-us/articles/12872879801105#section:RetrieveLicense).

Once you have the license file, it should be placed in one of the following default locations depending on your
operating system:

* `C:\gurobi\` for Windows
* `/opt/gurobi/` for Linux
* `/Library/gurobi/` for Mac OS X

Alternatively, you can set the `GRB_LICENSE_FILE` environment variable to the path of your license file. Doing so will
override the above default locations.

Please note that the code has been tested with Gurobi version 11, using a free academic license.

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

## Testing
To run unit tests, execute:
```
python3 utils/test_suite.py
```

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
│   ├── logger.py               # Utility for logging results
│   ├── plotter.py              # Utility for plotting results
│   └── test_suite.py           # Code for unit tests for the code
│
├── .gitignore                  # File specifying untracked files to ignore
├── README.md                   # Project description and documentation
├── requirements.txt            # List of dependencies for the project
└── main.py                     # Main script to run the optimizer
```

## Citing This Work

If you use this package in your research or for educational purposes, please cite the following publication:

> H. Ingimundardottir and T. P. Runarsson, "Evolving Submodels for Column Generation in Cutting and Packing for Glulam Production," 2024 IEEE Congress on Evolutionary Computation (CEC), Yokohama, Japan, 2024, pp. 1-8, doi: [10.1109/CEC60901.2024.10612070](https://doi.org/10.1109/CEC60901.2024.10612070).

### Example BibTeX Entry
```bibtex
@INPROCEEDINGS{Ingimundardottir2024,
  author={Ingimundardottir, Helga and Runarsson, Thomas Philip},
  booktitle={2024 IEEE Congress on Evolutionary Computation (CEC)}, 
  title={Evolving Submodels for Column Generation in Cutting and Packing for Glulam Production}, 
  year={2024},
  volume={},
  number={},
  pages={1-8},
  keywords={Pressing;Evolutionary computation;Production facilities;Manufacturing;Iterative methods;Feeds;Optimization;Glued laminated timber;column generation;evolutionary strategy;cutting and packing problems;real-world application},
  doi={10.1109/CEC60901.2024.10612070}
}
```
