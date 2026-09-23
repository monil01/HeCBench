JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -f hybridinput.txt hybridoutput.txt

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl r
