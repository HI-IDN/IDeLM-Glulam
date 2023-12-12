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
│   ├── __init__.py
│   ├── cutting_pattern.py      # Code for cutting pattern generation (Gurobi MIP)
│   └── packaging.py            # Code for packaging optimization (Gurobi MIP)
│
├── strategies/
│   ├── __init__.py
│   └── evolution_strategy.py   # Code for evolution strategy (1+1 ES)
│
├── utils/
│   ├── __init__.py
│   └── data_processor.py       # Utility for processing data file
│
└── main.py                     # Main script to run the optimizer
```