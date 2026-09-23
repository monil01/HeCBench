JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 8 1024 768 2000
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 8 1024 12288 2000
