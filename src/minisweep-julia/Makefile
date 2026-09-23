JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl --niterations 100
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl --niterations 1 --ncell_x 32 --ncell_y 32 --ncell_z 64 --ne 64 --na 32 --nblock_z 64
