JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__

run: main
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 1 8000 100
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 3 8000 100
