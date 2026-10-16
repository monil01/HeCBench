JULIA ?= julia
PROJECT ?= --project=../../_julia_ka_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__ result.raw

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl ../../affine-cuda/data/CT-MONO2-16-brain.raw result.raw 1000
