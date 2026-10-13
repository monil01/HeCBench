IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX = g++
HIPCC ?= hipcc
RUN_ARGS ?= -l 32 11 -t 4.7 0.1 -a 10 2 2000 10 -h 1.0 -z 7919
LAUNCHER ?=
all: main kernel.hip
main: main.cpp utils.cpp heap.cpp utils.h heap.h findex.h launch.h prng_host.h
	$(CXX) -O3 -std=c++17 -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include main.cpp utils.cpp heap.cpp -o $@ -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -lpthread -ldl
kernel.hip: kernel.hip.cpp kernel_prng.h kernel_metropolis.h kernel_reduction_hip.h utils.h
	$(HIPCC) -O3 --genco kernel.hip.cpp -o $@
kernel.ptx: kernel.cu kernel_prng.h kernel_metropolis.h kernel_reduction.h utils.h
	nvcc -O3 -ptx kernel.cu -o $@
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
