JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	@true

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl ../inversek2j-sycl/coord_in.txt 100
