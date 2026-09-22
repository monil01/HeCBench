JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -f *.dat

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 1024 16384 64 100
