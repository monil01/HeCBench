JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__

run: main
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 3 128 784 512 10 0 0 100
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 3 128 784 512 10 0 1 100
