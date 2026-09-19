JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 4096 2048 0 102
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 4096 2048 1 102
