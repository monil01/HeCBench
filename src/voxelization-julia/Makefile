JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__

test: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl ../voxelization-cuda/data/test/ 1

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl ../voxelization-cuda/data/test/ 100000
