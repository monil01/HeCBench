JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__ velocity_gpu.dat

run: main
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl
