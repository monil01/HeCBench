JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 8 4096 512 100
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 8 4096 2048 100
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 8 4096 5184 100
