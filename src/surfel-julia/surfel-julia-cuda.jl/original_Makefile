JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 1024 640 480 100
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 30000 640 480 100

test: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 128 64 48 2
