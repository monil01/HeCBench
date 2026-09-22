JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=
ARGS ?= -a 0.1 -w 1 -r 3 -m 1000 -n 999 -e 1

main:
	@true

clean:
	rm -f

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl $(ARGS)
