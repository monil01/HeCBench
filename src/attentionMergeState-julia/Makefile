JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__

run: main
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 1024 32 128 1000
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 1024 64 128 1000
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 1024 128 128 1000
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 1024 256 128 1000
