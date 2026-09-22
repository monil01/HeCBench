JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 4 1000 1
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 4 4000 0

test: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 2 2 1
