JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__

test: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 1 64 512 4096

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 1
