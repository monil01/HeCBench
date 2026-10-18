JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__ output.nrrd

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl -s 512
