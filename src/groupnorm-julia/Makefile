JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	@true

run: main
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 32 32 16 16 32 100
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 32 128 64 64 32 100
