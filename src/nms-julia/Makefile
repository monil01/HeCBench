JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -f output.txt

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl ../nms-cuda/detections.txt output.txt 1000 --check
