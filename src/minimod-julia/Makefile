JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__ snapshot.it*.raw

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl --warm-up --niters 1 --grid 256 --nsteps 1000
