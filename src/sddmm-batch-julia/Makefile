JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__

run: main
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 32 1024 1024 1024 1024 1000 1
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 32 4096 4096 4096 4096 1000 0
