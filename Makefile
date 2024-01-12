.PHONY: all single es

# Define the number of runs
NUM_RUNS := 10

# Create a sequence of runs
RUNS := $(shell seq 0 $(shell expr $(NUM_RUNS) - 1))

# Retrieve git tag, fall back to short commit hash if no tags are found
VERSION := $(shell git describe --tags --always --abbrev=0 2>/dev/null || git rev-parse --short HEAD)

# Define depths
DEPTHS := 90 115 140 160 185

# Main target for ES mode
all: es

# Define a rule for each depth that depends on its runs for ES mode
es:
	@$(foreach depth,$(DEPTHS), make $(addprefix es_, $(depth)) depth=$(depth);)

# Define a rule for each run for ES mode
$(addprefix es_, $(DEPTHS)):
	@$(foreach run,$(RUNS), \
		make data/$(VERSION)/soln_ES_d$(depth)_$(run).log depth=$(depth) run=$(run);)


data/$(VERSION)/soln_ES_d$(depth)_$(run).json:
	@echo "Running ES for depth $(depth) run $(run)"
	@mkdir -p data/$(VERSION)
	python3 main.py --mode ES --depth $(depth) --run $(run) --name $(VERSION) | tee $(@:.json=.log) 2>&1
