JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	@true

run: main
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl inputs/100k.txt 17 100 1
