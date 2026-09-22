JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 102400 1024 100

test: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 64 64 2
