JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 256 4096 100
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 256 8192 100
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 256 12288 100
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 16384 4096 100
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 16384 8192 100
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 16384 12288 100
