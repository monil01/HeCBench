JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -f *.dat

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl -v -s 128
