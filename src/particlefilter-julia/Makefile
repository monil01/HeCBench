JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__ output.txt

run: main
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl -x 128 -y 128 -z 10 -np 400000
