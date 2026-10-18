JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__ image_out.pgm

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 1000 0.5 502 458
