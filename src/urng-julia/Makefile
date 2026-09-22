JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -f output.bmp

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl ../urng-sycl/URNG_Input.bmp 1000
