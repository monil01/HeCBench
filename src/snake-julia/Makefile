JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__

run: main
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 100 ./Datasets/ERR240727_1_E2_30000Pairs.txt 30000 1000
