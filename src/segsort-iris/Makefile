IRIS_ROOT ?= /tmp/iris
CXX ?= g++
HIPCC ?= /opt/rocm/bin/hipcc
CXXFLAGS ?= -O3 -std=c++17
RUN_ARGS ?=
all: main kernel.hip
main: main.cpp gpu_lock.hpp
	$(CXX) $(CXXFLAGS) -I$(IRIS_ROOT)/include $< -o $@ -L$(IRIS_ROOT)/build/src/runtime -Wl,-rpath,$(IRIS_ROOT)/build/src/runtime -liris
kernel.hip: kernel.hip.cpp $(wildcard device/*.cuh)
	$(HIPCC) -O3 --genco $< -o $@
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip
