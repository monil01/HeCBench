JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__

run: main
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 4 12 256 256 1000
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 4 12 1024 1024 1000
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 16 12 1024 1024 1000
