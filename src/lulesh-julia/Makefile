JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__

run: main
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl -i 100 -s 128 -r 11 -b 1 -c 1
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl -i 100 -s 128 -r 11 -b 1 -c 1
