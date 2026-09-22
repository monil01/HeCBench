JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__

run: main
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl data/dirty_4096.img data/psf_4096.img 1000
