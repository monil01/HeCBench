JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	@true

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 100 0 ../urng-sycl/URNG_Input.bmp
