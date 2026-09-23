JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	rm -rf data

clean:
	rm -rf data

data/input_14_1_128.bin: main
	mkdir -p data
	touch data/input_14_1_128.bin

run: data/input_14_1_128.bin
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 0 100
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 1 100
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 2 100
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 3 100
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 4 100
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 5 100
