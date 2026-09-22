JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=
ARGS ?= 2 20 7

main:
	@true

clean:
	rm -f

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl $(ARGS)
