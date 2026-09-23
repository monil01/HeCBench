JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__ *.dat *.timing *.txt *.csv

run: main
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl -num_systems=30000
