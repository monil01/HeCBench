JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=
ARGS ?= -f ../rsc-cuda/input/vectors.csv -g 512 -r 10 -w 1

main:
	@true

clean:
	rm -rf __pycache__

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl $(ARGS)
