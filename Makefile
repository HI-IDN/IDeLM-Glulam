.PHONY: all single es stats logs

# Define the number of runs
NUM_RUNS := 10

# Create a sequence of runs
RUNS := $(shell seq 1 $(shell expr $(NUM_RUNS)))

# Retrieve git tag, fall back to short commit hash if no tags are found
VERSION := $(shell git describe --tags --always --abbrev=0 2>/dev/null || git rev-parse --short HEAD)
SLURMOUTPUT := $(shell pwd)/data/slurm

# Define depths
DEPTHS := 90 115 140 160 185 91

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

logs:
	@echo "Generate structured logs for all runs"
	@$(foreach log,$(wildcard $(SLURMOUTPUT)/*.err), \
		make $(log:.err=.log) --no-print-directory;)

stats:
	@echo "Generate objective and rho plots for all depths"
	@$(foreach depth,$(DEPTHS), \
		make data/$(VERSION)/ES_obj_d$(depth).png data/$(VERSION)/ES_rho_d$(depth).png DEPTH=$(depth) --no-print-directory;)
	@echo "Plot the best press for all runs"
	@$(foreach file,$(wildcard data/$(VERSION)/*.csv), \
		make $(file:.csv=.png) --no-print-directory;)
	@echo "Plot comparisons for all depths"
	make data/$(VERSION)/ES_run.png  --no-print-directory

%.png: %.csv utils/plot_press.R
	Rscript utils/plot_press.R $< $@

%/ES_obj_d$(DEPTH).png: JSON := $(wildcard data/$(VERSION)/soln_ES_d$(DEPTH)_*.json*)
%/ES_obj_d$(DEPTH).png: utils/plot_ES_obj.R
	Rscript utils/plot_ES_obj.R $@ $(JSON)

%/ES_rho_d$(DEPTH).png: JSON := $(wildcard data/$(VERSION)/soln_ES_d$(DEPTH)_*.json*)
%/ES_rho_d$(DEPTH).png: utils/plot_ES_rho.R
	Rscript utils/plot_ES_rho.R $@ $(JSON)

%/ES_run.png: JSON := $(wildcard data/$(VERSION)/soln_ES_d*.json*)
%/ES_run.png: utils/plot_ES_run.R logs
	Rscript utils/plot_ES_run.R $@ $(SLURMOUTPUT) $(JSON)

%.log: %.err utils/parse_log.py
	python3 utils/parse_log.py --raw_log $< --output_file $@
