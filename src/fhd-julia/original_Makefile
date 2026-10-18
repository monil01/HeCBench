JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__

test: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 1000 8192 1

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 100000 2097152 0
