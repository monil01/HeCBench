JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	@true

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl ../mis-cuda/internet.egr 100
