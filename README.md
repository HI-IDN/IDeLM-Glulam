```
IDeLM-Glulam/
│
├── config/
│   └── settings.py              # Configuration file for glulam dimensions and settings
│
├── data/
│   └── glulam.csv               # Default data file
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
└── main.py                     # Main script to run the optimizer
```