.PHONY: all single es stats

# Define the number of runs
NUM_RUNS := 10

# Create a sequence of runs
RUNS := $(shell seq 1 $(shell expr $(NUM_RUNS)))

# Retrieve git tag, fall back to short commit hash if no tags are found
VERSION := $(shell git describe --tags --always --abbrev=0 2>/dev/null || git rev-parse --short HEAD)
SLURMOUTPUT := $(shell pwd)/data/slurm

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
		make data/$(VERSION)/soln_ES_d$(depth)_$(run).json depth=$(depth) run=$(run);)


data/$(VERSION)/soln_ES_d$(depth)_$(run).json: $(FILE)
	@echo "Running ES for depth $(depth) run $(run)"
	@mkdir -p data/$(VERSION)
	python3 main.py --mode ES --depth $(depth) --run $(run) --name $(VERSION) \
		--file $(or $(FILE),$(error Missing data file))



stats:
	@echo "Generating stats"
	#make data/$(VERSION)/ES_run.png
	# for all csv files in data/$(VERSION) generate a plot
	@$(foreach file,$(wildcard data/$(VERSION)/*.csv), \
		make $(file:.csv=.png) --no-print-directory;)
	@$(foreach log,$(wildcard $(SLURMOUTPUT)/*.err), \
		make $(log:.err=.log) --no-print-directory;)

%.png: %.csv utils/plot_press.R
	@echo "Plotting $@"
	Rscript utils/plot_press.R $< $@

%/ES_run.png: JSON_FILES=$(wildcard $*/*.json*)
%/ES_run.png: utils/plot_ES.R $(JSON_FILES)
	Rscript utils/plot_ES.R $@ $(JSON_FILES)

%.log: %.err utils/parse_log.py
	python3 utils/parse_log.py --raw_log $< --output_file $@
