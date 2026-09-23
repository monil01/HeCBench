JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -f output.txt result*
	rm -rf __pycache__

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 10 20 256 65536 65536 1000 none output.txt 1
