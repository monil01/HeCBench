JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl -s 16384 -i 100 -x 512 -y 2
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl -s 16384 -i 100 -x 128 -y 8
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl -s 16384 -i 100 -x 32 -y 32
