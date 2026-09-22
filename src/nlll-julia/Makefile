JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 2048 1000 1000

test: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 256 128 2
