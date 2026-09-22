JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__

run: main
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 1 2 8 257 5
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 2 2 8 257 5
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 3 2 8 257 5
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 4 2 8 257 5
