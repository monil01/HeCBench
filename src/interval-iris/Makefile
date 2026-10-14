IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
RUN_ARGS ?= 0 100000
CXXFLAGS ?= -O3 -std=c++17
all: main kernel.hip
main: main.cpp interval.h diagnostics.h Makefile
	$(CXX) $(CXXFLAGS) -I../interval-cuda -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include main.cpp -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread -o $@
kernel.hip: kernel.hip.cpp gpu_interval.h hip_interval_lib.h hip_interval_rounded_arith.h interval.h
	$(HIPCC) -O3 --genco kernel.hip.cpp -o $@
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
	IRIS_ARCHS=HIP ./main 1 100000
clean:
	rm -f main kernel.hip kernel.ptx *.o
