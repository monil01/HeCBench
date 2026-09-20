JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 65536 2048 0 1000
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 65536 2048 1 1000
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 65536 2048 2 1000
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 65536 2048 3 1000
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 8192 8192 0 1000
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 8192 8192 1 1000
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 8192 8192 2 1000
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 8192 8192 3 1000
