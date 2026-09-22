JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=
ARGS ?= -f ../tqs-cuda/input/patternsNP100NB512FB25.txt -r 2 -w 1 -s 320 -q 320 -n 10

main:
	@true

clean:
	rm -rf __pycache__

run: main
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl $(ARGS)
