.PHONY: all runs

# Define the number of runs
NUM_RUNS := 10

# Create a sequence of runs
RUNS := $(shell seq 0 $(shell expr $(NUM_RUNS) - 1))

# Retrieve git tag, fall back to short commit hash if no tags are found
VERSION := $(shell git describe --tags --always --abbrev=0 2>/dev/null || git rev-parse --short HEAD)

# Define depths
DEPTHS := 90 115 140 160

# Main target
all: $(DEPTHS)

# Define a rule for each depth that depends on its runs
$(DEPTHS):
	@$(MAKE) --no-print-directory DEPTH=$@ runs

# Rule for each run
runs:
	@$(foreach run,$(RUNS),python3 main.py --depth $(DEPTH) --name $(VERSION) --run $(run);)

