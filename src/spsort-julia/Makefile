JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__

run: main
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 20082 20082 150616 1000
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 41490 41490 17590850 1000
