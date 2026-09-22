JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=
ARGS ?= 255 8 100 2

main:
	@true

clean:
	rm -rf __pycache__ input.bin output.bin

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl $(ARGS)
