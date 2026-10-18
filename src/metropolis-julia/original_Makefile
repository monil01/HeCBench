JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__ trials.dat

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl -l 32 11 -t 4.7 0.1 -a 10 2 2000 10 -h 1.0 -z 7919
