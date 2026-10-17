JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=
ARGS ?= 1 1 64 128 384 5

main:
	@true

clean:
	rm -f

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl $(ARGS)
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 2 1 64 128 384 5
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 3 1 64 128 384 5
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 4 1 64 128 384 5
