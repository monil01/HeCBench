JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__ result.raw

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl /home/imo/HeCBench/src/affine-cuda/data/CT-MONO2-16-brain.raw result.raw 1000
