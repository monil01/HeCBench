JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__ Output_task1.txt Output_task2.txt

run: main
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl
