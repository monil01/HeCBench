JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 32 1 6 32 32 5 100
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 32 6 16 14 14 5 100
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 32 96 256 26 26 5 100
