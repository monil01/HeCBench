JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__

test: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl -l 2 -u 4 -n 32 -r 1

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl
