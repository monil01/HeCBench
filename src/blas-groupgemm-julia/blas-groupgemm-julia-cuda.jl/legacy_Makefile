JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__

test: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 2 4 64 96 4

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 100 8 4096 14336 32
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 100 60 2048 1408 16
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 100 160 5120 1536 8
