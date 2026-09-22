JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__

run: main
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 64 64 64 128 5 1
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 128 128 128 256 5 0
