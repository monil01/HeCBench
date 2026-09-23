JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -f lena_std.dds teapot512_std.dds

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl ../dxtc2-sycl/data/lena_std.ppm ../dxtc2-sycl/data/lena_ref.dds 100
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl ../dxtc2-sycl/data/teapot512_std.ppm ../dxtc2-sycl/data/teapot512_ref.dds 100
