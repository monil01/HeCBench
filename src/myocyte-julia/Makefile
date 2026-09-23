JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

.PHONY: main clean run

main:
	@true

clean:
	rm -rf __pycache__ output.txt

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 100
